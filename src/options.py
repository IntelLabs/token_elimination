# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument("--warmup_steps", type=int, default=1000)
        self.parser.add_argument("--total_steps", type=int, default=1000)
        self.parser.add_argument(
            "--scheduler_steps",
            type=int,
            default=None,
            help="total number of step for the scheduler, if None then scheduler_total_step = total_step",
        )
        self.parser.add_argument("--accumulation_steps", type=int, default=1)
        self.parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
        self.parser.add_argument("--optim", type=str, default="adam")
        self.parser.add_argument("--scheduler", type=str, default="fixed")
        self.parser.add_argument("--weight_decay", type=float, default=0.1)
        self.parser.add_argument("--fixed_lr", action="store_true")

    def add_eval_options(self):
        self.parser.add_argument(
            "--write_results",
            type=str2bool,
            nargs="?",
            default=False,
            help="Save the results of to a file, mainly for later submitting them.",
        )
        self.parser.add_argument(
            "--write_crossattention_scores",
            type=str2bool,
            nargs="?",
            default=False,
            help="save dataset with cross-attention scores",
        )

    def add_reader_options(self):
        self.parser.add_argument(
            "--train_data", type=str, default="none", help="path of train data"
        )
        self.parser.add_argument("--eval_data", type=str, default="none", help="path of eval data")
        self.parser.add_argument("--model_size", type=str, default="base")
        self.parser.add_argument("--model_name", type=str, default=None)
        self.parser.add_argument(
            "--use_checkpoint",
            type=str2bool,
            nargs="?",
            default=True,
            help="use checkpoint in the encoder",
        )
        self.parser.add_argument(
            "--text_maxlength",
            type=int,
            default=200,
            help="maximum number of tokens in text segments (question+passage)",
        )
        self.parser.add_argument(
            "--answer_maxlength",
            type=int,
            default=100,
            help="maximum number of tokens used to train the model, no truncation if -1",
        )
        self.parser.add_argument(
            "--no_title", type=int, default=0, help="article titles not included in passages"
        )
        self.parser.add_argument(
            "--n_context", type=int, default=1, help="The number of contexts to use."
        )

    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument(
            "--name", type=str, default="experiment_name", help="name of the experiment"
        )
        self.parser.add_argument(
            "--checkpoint_dir", type=str, default="./checkpoint/", help="models are saved here"
        )
        self.parser.add_argument(
            "--model_path", type=str, default="none", help="path for retraining"
        )

        # dataset parameters
        self.parser.add_argument(
            "--per_gpu_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training."
        )
        self.parser.add_argument("--maxload", type=int, default=-1)

        self.parser.add_argument(
            "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
        )
        self.parser.add_argument(
            "--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)"
        )
        self.parser.add_argument(
            "--seed", type=int, default=0, help="random seed for initialization"
        )
        # training parameters
        self.parser.add_argument(
            "--eval_freq",
            type=int,
            default=500,
            help="evaluate model every <eval_freq> steps during training",
        )
        self.parser.add_argument(
            "--save_freq",
            type=int,
            default=5000,
            help="save model every <save_freq> steps during training",
        )
        self.parser.add_argument(
            "--eval_print_freq",
            type=int,
            default=10,
            help="print intermdiate results of evaluation every <eval_print_freq> steps",
        )

        self.parser.add_argument(
            "--pad_to_max_length",
            type=str2bool,
            nargs="?",
            default=False,
            help="Should the passages all be padded to the max length provided (True value here), or to the max length of the longest passage? (False value here).",
        )

        self.parser.add_argument(
            "--overwrite_checkpoint",
            type=int,
            default=0,
            help="Force overwrite of the checkpoint directory",
        )
        self.parser.add_argument("--gpus", type=int, default=1, help="number of accelerators")

        self.parser.add_argument(
            "--fp16", type=str2bool, nargs="?", default=False, help="enable fp16 training"
        )
        self.parser.add_argument(
            "--bf16", type=str2bool, nargs="?", default=True, help="enable bf16 training"
        )
        self.parser.add_argument(
            "--ampere", type=str2bool, nargs="?", default=False, help="enable ampere tf32"
        )

        self.parser.add_argument(
            "--wandb_project_name",
            type=str,
            default="fid",
            help="Name of Weights and Biases project.",
        )
        self.parser.add_argument(
            "--wandb_group_name",
            type=str,
            default="early",
            help="Name of Weights and Biases group.",
        )
        self.parser.add_argument(
            "--wandb_run_name", type=str, default="ee_conf", help="Name of Weights and Biases name."
        )

        self.parser.add_argument(
            "--use_eval_example_count",
            type=int,
            default=-1,
            help="Number of examples to use during evaluation.",
        )

        self.parser.add_argument(
            "--measure_flops",
            type=str2bool,
            nargs="?",
            default=False,
            help="Whether to measure the FLOPS (macs) or not.",
        )

        self.parser.add_argument(
            "--answer_minlength", type=int, default=10, help="Minimum answer length."
        )
        self.parser.add_argument(
            "--num_beams", type=int, default=4, help="Number of beams for beam search."
        )
        self.parser.add_argument(
            "--decoder_early_exit_type",
            type=str,
            default=None,
            choices=["classifier", "softmax", "state"],
            help="""
The type of confidence estimation method to use:
"classifier": A classifier model to estimate the confidence, given the hidden states at the current layer.
"softmax": Applies the softmax operation to over the vocabulary prediction at the current layer. If the confidence is then the difference between the first and second highest ranked tokens.
"state": Computes the cosine similarity between the current hidden states and the previous ones.
        """,
        )
        self.parser.add_argument(
            "--decoder_early_exit_thres",
            type=float,
            default=None,
            help="The confidence threshold for the current method. If the confidence value exceeds this value, an early exit is triggered.",
        )

        self.parser.add_argument(
            "--freeze",
            type=str,
            default=None,
            help="Freeze the weights of the model that contain the string specified here, during training.",
        )

        self.parser.add_argument(
            "--train_conf_heads",
            type=str2bool,
            nargs="?",
            default=False,
            help="Whether to train confidence heads per layer or not.",
        )

        self.parser.add_argument(
            "--share_conf_heads",
            type=str2bool,
            nargs="?",
            default=False,
            help="Use a shared confidence estimation head for all the layers (True), or create separate heads per layer (False).",
        )

        self.parser.add_argument(
            "--conf_head_type",
            type=str,
            default="simple",
            help="maximum number of tokens used to train the model, no truncation if -1",
        )

        self.parser.add_argument(
            "--use_shared_decoder_lm_head",
            type=str2bool,
            nargs="?",
            default=False,
            help="Use a shared decoder head for all the layers (True), or create separate heads per layer (False).",
        )

        self.parser.add_argument(
            "--decoder_early_exit_tau",
            type=int,
            default=-1,
            help="The tau in the confidence threshold scheduling mechanism, which modifies the rate of change.",
        )
        self.parser.add_argument(
            "--decoder_early_exit_alpha",
            type=float,
            default=-1,
            help="The alpha in the confidence threshold scheduling mechanism, which modifies the coefficient of the scheduling component.",
        )
        self.parser.add_argument("--dev_metric", type=str, default="rouge", help="dev_metric")

        self.parser.add_argument(
            "--infer_bf16",
            type=str2bool,
            nargs="?",
            default=False,
            help="Use BF16 during inference.",
        )

        self.parser.add_argument(
            "--use_long_version",
            type=str2bool,
            nargs="?",
            default=False,
            help="If checked, then the second answer in the answer list is selected, instead of the first one (i.e. answers[1]). This is meant for the MS MARCO dataset files, where the longer version of the answer is present.",
        )

        self.parser.add_argument(
            "--filter_to_take",
            type=int,
            default=-1,
            help="Amount of tokens to keep after filtering.",
        )
        self.parser.add_argument(
            "--filter_to_take_percent",
            type=float,
            default=-1,
            help="Percentage of tokens to keep after filtering.",
        )
        self.parser.add_argument(
            "--filter_token",
            type=int,
            default=-1,
            help="The generated token index, where filtering needs to occur.",
        )
        self.parser.add_argument(
            "--filter_layer",
            type=int,
            default=-1,
            help="The decoder layer index, where filtering needs to occur.",
        )

        self.parser.add_argument(
            "--filter",
            type=str2bool,
            nargs="?",
            default=False,
            help="Whether Token Filtering should be used or not.",
        )
        self.parser.add_argument(
            "--filter_use_values",
            type=str2bool,
            nargs="?",
            default=False,
            help="Whether the normalization with the values tensor should be used or not.",
        )
        self.parser.add_argument(
            "--filter_use_last_state",
            type=str2bool,
            nargs="?",
            default=True,
            help="Whether to use only the latest layer in the cross-attention score computation or not.",
        )

        self.parser.add_argument(
            "--setup_test_file",
            type=str,
            default=None,
            help="The file containing the interval configurations.",
        )
        self.parser.add_argument(
            "--setup_test_index",
            type=int,
            default=0,
            help="The interval configuration to apply. For example, if the interval configuration file contains 30 entries, and the setup_test_index=4 then the fifth configuration will override all the test time arguments specified in it.",
        )

    def print_options(self, opt):
        message = "\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f"\t(default: {default_value})"
            message += f"{str(k):>30}: {str(v):<40}{comment}\n"

        expr_dir = Path(opt.checkpoint_dir) / opt.name
        model_dir = expr_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(expr_dir / "opt.log", "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "-1"
        opt.local_rank = int(os.environ["LOCAL_RANK"])  # required for torchrun
        return opt


def get_options(use_reader=False, use_optim=False, use_eval=False):
    options = Options()
    if use_reader:
        options.add_reader_options()
    if use_optim:
        options.add_optim_options()
    if use_eval:
        options.add_eval_options()
    return options.parse()
