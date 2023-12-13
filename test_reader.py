# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import wandb
from deepspeed.profiling.flops_profiler import FlopsProfiler
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import src.data
import src.evaluation
import src.kilt_evaluation
import src.model
import src.slurm
import src.util
from src.options import Options


def start_timer():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    return time.time()


def end_timer():
    torch.cuda.synchronize()
    return time.time()


def log_results(metrics):
    if wandb.run is not None:
        wandb.log(metrics)


def evaluate(model, dataset, dataloader, tokenizer, opt):
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    if opt.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage()
    total = 0
    exactmatch = []
    if opt.write_results:
        write_path = (
            Path(opt.checkpoint_dir) / f"{opt.wandb_group_name}_test_results_{opt.current_run_hash}"
        )
        fw = open(f"{write_path}.txt", "a")

    infer_times = []
    all_flops = []
    all_context_counts = []
    all_eval_scores = []

    kilt_examples = []
    kilt_ans = []

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            (idx, _, _, context_ids, context_mask) = batch
            logger.info(f"context_ids: {context_ids.shape if context_ids is not None else None}")

            if opt.write_crossattention_scores:
                model.reset_score_storage()

            all_context_counts.append(int(context_ids.shape[1]))

            inference_start_time = start_timer()

            if opt.measure_flops > 0:
                prof = FlopsProfiler(model)
                prof.start_profile()

            outputs = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                num_beams=opt.num_beams,
                no_repeat_ngram_size=2,
                min_length=opt.answer_minlength,
                max_length=opt.answer_maxlength,
                early_stopping=True,
            )

            if opt.measure_flops > 0:
                prof.stop_profile()
                flops = prof.get_total_flops(as_string=False)
                prof.end_profile()
                del prof

                all_flops.append(flops)

            inference_end_time = end_timer()
            inference_time = inference_end_time - inference_start_time
            infer_times.append(inference_time)

            if opt.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(context_mask.cuda())

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                example = dataset.data[idx[k]]

                kilt_ans.append(ans)
                kilt_examples.append(example)

                if "answers" in example:
                    score = src.evaluation.ems(ans, example["answers"])
                    current_eval_scores = src.evaluation.get_max_rouge(
                        ans, example["answers"], scorer
                    )

                    exactmatch.append(score)
                    all_eval_scores.append(current_eval_scores)

                if opt.write_results:
                    fw.write(str(example["id"]) + "\t" + ans + "\n")

                if opt.write_crossattention_scores:
                    for j in range(context_ids.size(1)):
                        example["ctxs"][j]["score"] = crossattention_scores[k, j].item()

                total += 1
            if (i + 1) % opt.eval_print_freq == 0:
                log = f"Process rank:{opt.global_rank}, {i+1} / {len(dataloader)}"
                if len(exactmatch) == 0:
                    log += "| no answer to compute scores"
                else:
                    log += f" | average = {np.mean(exactmatch):.3f}"
                logger.warning(log)

            model.decoder.filter_hidden_states = False
            if hasattr(model.decoder, "decoder_passage_mask"):
                del model.decoder.decoder_passage_mask
            model.decoder.decoder_passage_mask = None

    logger.warning(
        f"Process rank:{opt.global_rank}, total {total} | average = {np.mean(exactmatch):.3f}"
    )
    if opt.is_distributed:
        torch.distributed.barrier()
    score, total = src.util.weighted_average(np.mean(exactmatch), total, opt)

    metrics = {
        "inference_time_mean": np.mean(infer_times),
        "context_count_mean": np.mean(all_context_counts),
        "flops_mean": np.mean(all_flops),
        "decoder_exit_layer_index_mean": np.mean(model.decoder.ee_layers)
        if len(model.decoder.ee_layers) > 0
        else -1,
    }

    all_eval_scores_dict = pd.DataFrame(all_eval_scores).mean().to_dict()

    metrics.update(all_eval_scores_dict)
    metrics["em"] = score

    kilt_eval_result = src.kilt_evaluation.evaluate_from_answers(kilt_examples, kilt_ans)
    metrics["kilt_rougel"] = kilt_eval_result["downstream"]["rougel"]
    metrics["kilt_f1"] = kilt_eval_result["downstream"]["f1"]

    logger.info(f"metrics: {metrics}")

    log_results(metrics)

    return score, total


def hash_dict(d):
    return str(hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    opt = options.parse()

    if opt.setup_test_file is not None:
        with open(opt.setup_test_file, "r") as f:
            setup_data = json.load(f)

        setup_data_current = setup_data[opt.setup_test_index]
        for k_setup, v_setup in setup_data_current.items():
            org_types = [
                action.type
                for action in options.parser._actions
                if action.option_strings[0].replace("--", "") == k_setup
            ]
            if len(org_types) == 0:
                continue
            org_type = org_types[0]
            # ensure the type of the setup matches the expected type
            setattr(opt, k_setup, org_type(v_setup))
            get_opt_attr = getattr(opt, k_setup)
            print(f"new {k_setup}: {get_opt_attr}, type: {type(get_opt_attr)}")

    if opt.write_results:
        opt.checkpoint_dir = opt.model_path

    if opt.filter_to_take_percent != -1:
        opt.filter_to_take = int(opt.filter_to_take_percent * opt.n_context)

    opt.current_run_hash = hash_dict(opt.__dict__)

    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()
    opt.train_batch_size = opt.per_gpu_batch_size * max(1, opt.world_size)

    dir_path = Path(opt.checkpoint_dir) / opt.name
    directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    dir_path.mkdir(parents=True, exist_ok=True)
    if opt.write_results:
        (dir_path / "test_results").mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(
        opt.is_main, opt.is_distributed, Path(opt.checkpoint_dir) / opt.name / "run.log"
    )

    try:
        wandb.init(
            project=opt.wandb_project_name,
            group=opt.wandb_group_name,
            name=opt.wandb_run_name,
            dir="/tmp",
        )
        wandb.config.update(opt)
    except Exception as e:
        logger.error(f"WANDB did not initialize correctly: {e}")

    if not directory_exists and opt.is_main:
        options.print_options(opt)

    # Load the tokenizer
    tokenizer = transformers.T5Tokenizer.from_pretrained("t5-base", return_dict=False)

    # Load the collator
    collator_function = src.data.Collator(
        opt.text_maxlength, tokenizer, pad_to_max_length=opt.pad_to_max_length, opt=opt
    )

    # Load evaluation examples
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,  # use the global rank and world size attibutes to split the eval set on multiple gpus
        world_size=opt.world_size,
    )

    if opt.use_eval_example_count != -1:
        eval_examples = eval_examples[: opt.use_eval_example_count]

    # Create the dataset and sampler
    eval_dataset = src.data.Dataset(
        eval_examples, opt.n_context, opt=opt, collator=collator_function
    )

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        num_workers=0,
        collate_fn=collator_function,
    )

    # Load the model
    model_class = src.model.FiDT5
    model = model_class.from_pretrained(opt.model_path)
    model = model.to(opt.device)
    if opt.infer_bf16:
        model = model.to(torch.bfloat16)

    # Insert the model configurations from the run arguments
    model.decoder.config.decoder_early_exit_type = opt.decoder_early_exit_type
    model.decoder.config.decoder_early_exit_thres = opt.decoder_early_exit_thres
    model.decoder.config.decoder_early_exit_tau = opt.decoder_early_exit_tau
    model.decoder.config.decoder_early_exit_alpha = opt.decoder_early_exit_alpha
    model.decoder.config.answer_maxlength = opt.answer_maxlength

    model.decoder.config.n_context = opt.n_context
    model.decoder.config.filter_to_take = opt.filter_to_take
    model.decoder.config.filter_token = opt.filter_token
    model.decoder.config.filter_layer = opt.filter_layer
    model.decoder.config.filter = bool(opt.filter)

    model.decoder.config.filter_use_values = bool(opt.filter_use_values)

    model.decoder.config.filter_use_last_state = bool(opt.filter_use_last_state)

    model.decoder.config.filter_to_take_percent = opt.filter_to_take_percent

    # softmax method uses the lm head during hidden states loop
    if opt.decoder_early_exit_type is not None and opt.decoder_early_exit_type == "softmax":
        model.decoder.get_lm_logits_output = model.get_lm_logits_output

    logger.info("Start eval")
    exactmatch, total = evaluate(model, eval_dataset, eval_dataloader, tokenizer, opt)

    logger.info(f"EM {100*exactmatch:.2f}, Total number of example {total}")

    if opt.write_results and opt.is_main:
        glob_path = Path(opt.checkpoint_dir) / opt.name / "test_results"
        write_path = Path(opt.checkpoint_dir) / opt.name / "final_output.txt"
        src.util.write_output(glob_path, write_path)
    if opt.write_crossattention_scores:
        src.util.save_distributed_dataset(eval_dataset.data, opt)
