# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import random

import torch
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        n_context=None,
        question_prefix="question:",
        title_prefix="title:",
        passage_prefix="context:",
        opt=None,
        collator=None,
    ):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()
        self.opt = opt
        self.collator = collator

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if "target" in example:
            target = example["target"]
            return target
        elif "answers" in example:
            if self.opt.use_long_version and len(example["answers"]) > 1:
                return example["answers"][1]
            return random.choice(example["answers"])
        else:
            return None

    def load_from_file(self, index):
        with open(self.data[index], "r") as f:
            example = json.load(f)

        for i_c, c in enumerate(example["ctxs"]):
            c["score"] = 1 / (i_c + 1)
        return example

    def __getitem__(self, index):
        if type(self.data[0]) == str:
            example = self.load_from_file(index)
        else:
            example = self.data[index]

        question = self.question_prefix + " " + example["question"]
        target = self.get_target(example)

        if "ctxs" in example and self.n_context is not None:
            contexts = example["ctxs"][: self.n_context]

            if self.opt.no_title:
                f = self.passage_prefix + " {}"
                passages = [f.format(c["text"]) for c in contexts]
            else:
                f = self.title_prefix + " {} " + self.passage_prefix + " {}"
                passages = [f.format(c["title"], c["text"]) for c in contexts]

            scores = [float(c["score"]) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None

        return {
            "index": index,
            "question": question,
            "target": target,
            "passages": passages,
            "scores": scores,
        }

    def sort_data(self):
        if type(self.data[0]) == str:
            return
        if self.n_context is None or not "score" in self.data[0]["ctxs"][0]:
            return
        for ex in self.data:
            ex["ctxs"].sort(key=lambda x: float(x["score"]), reverse=True)

    def get_example(self, index):
        return self.data[index]


def get_padded_tensor(ten_list, value=0):
    max_len = max([x.shape[1] for x in ten_list])
    padded_list = []
    for tensor in ten_list:
        if tensor.shape[1] < max_len:
            tensor = F.pad(
                input=tensor, pad=(0, max_len - tensor.shape[1], 0, 0), mode="constant", value=value
            )
        padded_list.append(tensor)
    return padded_list


def encode_passages(batch_text_passages, tokenizer, max_length, pad_to_max_length):
    # if padding to the max length, no padding to max passage length, and vice versa.
    padding = True if not pad_to_max_length else "max_length"

    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer(
            text_passages,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
        )
        passage_ids.append(p["input_ids"])
        passage_masks.append(p["attention_mask"])

    passage_ids = get_padded_tensor(passage_ids, value=0)
    passage_masks = get_padded_tensor(passage_masks, value=0)

    passage_ids = torch.stack(passage_ids)
    passage_masks = torch.stack(passage_masks)
    return passage_ids, passage_masks.bool()


class NoPooling(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features):
        return {"sentence_embedding": features["token_embeddings"]}


class Collator(object):
    def __init__(
        self, text_maxlength, tokenizer, answer_maxlength=20, pad_to_max_length=False, opt=None
    ):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength
        self.pad_to_max_length = pad_to_max_length
        self.opt = opt

    def __call__(self, batch):
        try:
            assert batch[0]["target"] != None
            index = torch.tensor([ex["index"] for ex in batch])
            target = [ex["target"] for ex in batch]
            target = self.tokenizer(
                target,
                max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
                padding=True if not self.pad_to_max_length else "max_length",
                return_tensors="pt",
                truncation=True if self.answer_maxlength > 0 else False,
            )
            target_ids = target["input_ids"]
            target_mask = target["attention_mask"].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            def append_question(example):
                if example["passages"] is None:
                    return [example["question"]]
                return [example["question"] + " " + t for t in example["passages"]]

            text_passages = [append_question(example) for example in batch]
            passage_ids, passage_masks = encode_passages(
                text_passages, self.tokenizer, self.text_maxlength, self.pad_to_max_length
            )

            return (index, target_ids, target_mask, passage_ids, passage_masks)
        except Exception as e:
            logging.error(f"Error in Collator: {e}")
            return (None, None, None, None, None)


def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith("traindir"):
        dir_files = os.listdir(data_path)
        examples = []
        for k, x in enumerate(dir_files):
            if global_rank > -1 and not k % world_size == global_rank:
                continue
            examples.append(f"{data_path}/{x}")
        return examples

    if data_path.endswith(".jsonl"):
        data = open(data_path, "r")
    elif data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        if global_rank > -1 and not k % world_size == global_rank:
            continue
        if data_path is not None and data_path.endswith(".jsonl"):
            example = json.loads(example)
        if not "id" in example:
            example["id"] = k
        for c in example["ctxs"]:
            if not "score" in c:
                c["score"] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith(".jsonl"):
        data.close()

    return examples


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, title_prefix="title:", passage_prefix="context:"):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + self.passage_prefix + " " + example[1]
        return example[0], text


class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True,
        )
        text_ids = encoded_batch["input_ids"]
        text_mask = encoded_batch["attention_mask"].bool()

        return index, text_ids, text_mask
