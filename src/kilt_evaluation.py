# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json

# utility to get gold answers
import pprint
import re
import string
from collections import Counter

from rouge import Rouge
from tqdm import tqdm


def get_gold_answers(gold):
    ground_truths = set()
    for item in gold["output"]:
        if "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())
        else:
            ground_truths.add("NOANS")
    return ground_truths


# utility to get gold titles
def get_gold_titles(gold):
    titles = set()
    for item in gold["output"]:
        if "provenance" in item:
            for provenance in item["provenance"]:
                if (
                    "title" in provenance
                    and provenance["title"]
                    and len(provenance["title"].strip()) > 0
                ):
                    titles.add(provenance["title"].strip())
    return titles


# utility to get max
def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


# F1 score definition
def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# EM score definition
def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ROUGEL score definition
def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def _calculate_metrics(gold_records, guess_records):
    assert len(gold_records) == len(guess_records), "different size gold: {} guess: {}".format(
        len(gold_records), len(guess_records)
    )

    total_count = 0

    # downstream metrics
    accuracy = 0
    normalized_em = 0
    normalized_f1 = 0
    rougel = 0

    # kilt metrics
    kilt_accuracy = 0
    kilt_em = 0
    kilt_f1 = 0
    kilt_rougel = 0

    for guess_item, gold_item in tqdm(zip(guess_records, gold_records)):
        # check ids
        assert (
            str(gold_item["id"]).strip() == str(guess_item["id"]).strip()
        ), "Items must have same order with same IDs"

        total_count += 1
        # check if each output of guess file exist in set of candidate answers
        gold_candidate_answers = get_gold_answers(gold_item)

        conditions = (len(guess_item["output"]) == 1) and ("answer" in guess_item["output"][0])
        assert conditions, f"you should provide exactly one valid answer for {guess_item['id']}"
        guess_answer = str(guess_item["output"][0]["answer"]).strip()

        if len(guess_answer) == 0:
            # empty answer
            continue

        # 0. accuracy = strict exact match
        local_accuracy = 0
        if guess_answer in gold_candidate_answers:
            local_accuracy = 1
        accuracy += local_accuracy

        # 1. normalized exact match
        local_em = _metric_max_over_ground_truths(
            _exact_match_score, guess_answer, gold_candidate_answers
        )
        normalized_em += local_em

        # 2. normalized f1
        local_f1 = _metric_max_over_ground_truths(_f1_score, guess_answer, gold_candidate_answers)
        normalized_f1 += local_f1

        # 3. rougel
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, guess_answer, gold_candidate_answers
        )
        rougel += local_rougel

    if total_count > 0:
        accuracy /= total_count
        normalized_em /= total_count
        normalized_f1 /= total_count
        rougel /= total_count
        kilt_accuracy /= total_count
        kilt_em /= total_count
        kilt_f1 /= total_count
        kilt_rougel /= total_count

    return {
        "kilt": {
            "KILT-accuracy": kilt_accuracy,
            "KILT-em": kilt_em,
            "KILT-f1": kilt_f1,
            "KILT-rougel": kilt_rougel,
        },
        "downstream": {
            "accuracy": accuracy,
            "em": normalized_em,
            "f1": normalized_f1,
            "rougel": rougel,
        },
    }


def validate_input(gold_records, guess_records):
    if len(gold_records) != len(guess_records):
        print(
            "WARNING: DIFFERENT SIZE gold: {} guess: {}".format(
                len(gold_records), len(guess_records)
            )
        )

    # align order
    gold_ids = []
    for gold in gold_records:
        assert str(gold["id"]).strip() not in gold_ids, "Gold IDs should be unique"
        gold_ids.append(str(gold["id"]).strip())

    id2guess_record = {}
    for guess in guess_records:
        assert str(guess["id"]).strip() not in id2guess_record, "Prediction IDs should be unique"
        id2guess_record[str(guess["id"]).strip()] = guess

    guess_records = []
    for id in gold_ids:
        if id in id2guess_record:
            guess_records.append(id2guess_record[id])
        else:
            raise ValueError("ERROR: no prediction provided for id: {}".format(id))

    return gold_records, guess_records


def load_data(filename):
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def evaluate(gold_records, guess_records):
    pp = pprint.PrettyPrinter(indent=4)

    # 0. validate input
    gold_records, guess_records = validate_input(gold_records, guess_records)

    # 1. downstream + kilt
    result = _calculate_metrics(gold_records, guess_records)

    pp.pprint(result)
    return result


def create_json_entry(jid, input_text, answers):
    return {
        "id": jid,
        "input": input_text,
        "output": [
            {
                "answer": answer,
                "provenance": [
                    {"wikipedia_id": "10593264", "wikipedia_title": "Therefore sign"},
                ],
            }
            for answer in answers
        ],
    }


def create_records(test_dataset, predicted_answers):
    gold_records = []
    guess_records = []

    for i in range(len(test_dataset)):
        example = test_dataset[i]
        predicted_answer = predicted_answers[i]

        gold_records.append(create_json_entry(i, example["question"], example["answers"]))
        guess_records.append(create_json_entry(i, example["question"], [predicted_answer]))
    return gold_records, guess_records


def evaluate_from_answers(test_dataset, predicted_answers):
    gold_records, guess_records = create_records(test_dataset, predicted_answers)

    return evaluate(gold_records, guess_records)
