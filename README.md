# Optimizing Retrieval-augmented Reader Models via Token Elimination

This repository contains the implementation of the method presented in the paper "Optimizing Retrieval-augmented Reader Models via Token Elimination".

The repository is based on the repository [facebookresearch/FiD](https://github.com/facebookresearch/FiD.git).

# Data

## Download data

In order to download the data, we provide the following description of how to obtain each one:

* *Data Corpus*: We used the entirety of the [kilt_wikipedia](https://huggingface.co/datasets/kilt_wikipedia) document collection. In each entry, we divide all the paragraphs into 100-word long passages, and later insert them into a standard Elasticsearch index.

* *ELI5*: We utilize the dataset [kilt_tasks/eli5](https://huggingface.co/datasets/kilt_tasks). Our development set we used is the last 3000 entries in the *train* subset provided, with the rest serving as a training set. The validation set, containing 1507 entries, serves as our *test* set. In addition, we submit to the [KILT Benchmark](https://eval.ai/web/challenges/challenge-page/689/leaderboard/1908).
* *MS MARCO*: For the train and dev datasets, we utilize the dataset provided in the [MS MARCO Passage Retrieval Page](https://microsoft.github.io/msmarco/), with the link provided [here](https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv). The dev set consists of the last 3000 entries in the queries list given above.
For the test set, we used the [ir_datasets](https://ir-datasets.com/) package, containing 6980 queries. Use the following code to download the test set:

```python
import ir_datasets
dataset = ir_datasets.load("msmarco-passage/dev/small")
for query in dataset.queries_iter():
    query # namedtuple<query_id, text>
```

* *NQ*: We utilize the dataset [Tevatron/wikipedia-nq](https://huggingface.co/datasets/Tevatron/wikipedia-nq). Our development set we used is the last 3000 entries in the *train* subset provided, with the rest serving as a training set. The validation set, containing 6.49k entries, serves as our *test* set.

## Retrieval For each Dataset

First, we create an Elasticsearch passage index over the entire the *Data Corpus* specified above.
Then, for any question from any of the datasets above, we use the Elasticsearch query operation over the passage index, to retrieve 250 passages. Then, we re-rank the passages using a [sentence transformer](sentence-transformers/multi-qa-mpnet-base-dot-v1) model, and keep only the top 100 ranked passages.

## Data format

The expected data format for each entry example is a dictionary containing:
- `id`: example id, optional
- `question`: question text
- `target`: answer used for model training, if not given, the target is randomly sampled from the 'answers' list
- `answers`: list of answer text for evaluation, also used for training if target is not given
- `ctxs`: a list of passages where each item is a dictionary containing
        - `title`: article title
        - `text`: passage text

Entry example:
```
{
  'id': '0',
  'question': 'What element did Marie Curie name after her native land?',
  'target': 'Polonium',
  'answers': ['Polonium', 'Po (chemical element)', 'Po'],
  'ctxs': [
            {
                "title": "Marie Curie",
                "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
            },
            {
                "title": "Marie Curie",
                "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
            }
          ]
}
```

# Pretrained models.

We include pretrained models, including both the standard FiD model, and the model after Early-Exit training:

|          | Standard FiD | Early-Exit |
|----------|--------------|------------|
| ELI5     | [Intel/eli5_fid](https://huggingface.co/Intel/eli5_fid)            | [Intel/eli5_fid_early_exit](https://huggingface.co/Intel/eli5_fid_early_exit)          |
| MS MARCO | [Intel/msmarco_fid](https://huggingface.co/Intel/msmarco_fid)            | [Intel/msmarco_fid_early_exit](https://huggingface.co/Intel/msmarco_fid_early_exit)          |
| NQ       | [Intel/nq_fid_lfqa](https://huggingface.co/Intel/nq_fid_lfqa)            | [Intel/nq_fid_lfqa_early_exit](https://huggingface.co/Intel/nq_fid_lfqa_early_exit)          |


# Training


we offer a training script [`train_reader.py`](train_reader.py) to train the FiD model.
For standard FiD training, use the command as follows:

```shell
torchrun --nproc-per-node=8 train_reader.py  \
--checkpoint_dir /tmp/out \
--model_size base  \
--train_data LONGNQ/train.json  \
--eval_data LONGNQ/dev.json  \
--use_checkpoint \
--lr 0.00005  \
--optim adamw  \
--scheduler linear  \
--weight_decay 0.01  \
--text_maxlength 256  \
--per_gpu_batch_size 1  \
--n_context 100  \
--total_step 60000  \
--warmup_step 1000  \
--name my_experiment_1  \
--eval_freq 5000  \
--save_freq 60000  \
--accumulation_steps 8  \
--use_eval_example_count 200  \
--bf16 \
--answer_minlength 100  \
--answer_maxlength 300
```

For Early-Exit training, we separate the training into 2 stages:

## Stage-I

To train the decoder layers for Early-Exiting, we first train using the following command:

```shell
torchrun --nproc-per-node=8 train_reader.py  \
--checkpoint_dir /tmp/stage1  \
--model_size base  \
--train_data LONGNQ/train.json  \
--eval_data LONGNQ/dev.json  \
--use_checkpoint \
--lr 0.00005  \
--optim adamw  \
--scheduler linear  \
--weight_decay 0.01  \
--text_maxlength 256  \
--per_gpu_batch_size 1  \
--n_context 100  \
--total_step 60000  \
--warmup_step 1000  \
--name my_experiment_1  \
--eval_freq 5000  \
--save_freq 60000  \
--accumulation_steps 8  \
--use_eval_example_count 200  \
--bf16 \
--use_shared_decoder_lm_head \
--answer_minlength 100  \
--answer_maxlength 300
```

## Stage-II

Once the model is trained per layer, we train the confidence head, for layer confidence estimation using the following:

```shell
python train_reader.py  \
--checkpoint_dir /tmp/stage2 \
--train_data LONGNQ/train.json  \
--eval_data LONGNQ/dev.json  \
--use_checkpoint \
--lr 0.00005  \
--optim adamw  \
--scheduler linear  \
--weight_decay 0.01  \
--text_maxlength 256  \
--per_gpu_batch_size 1  \
--n_context 100  \
--total_step 20000  \
--warmup_step 1000  \
--name my_experiment_1  \
--eval_freq 2500  \
--save_freq 20000  \
--accumulation_steps 1  \
--use_shared_decoder_lm_head 1  \
--bf16 \
--model_path PATH_TO_STAGE_1_MODEL  \
--model_size base  \
--train_conf_heads \
--freeze confidence_head_estimator  \
--share_conf_heads \
--conf_head_type 2layer  \
--answer_minlength 100  \
--answer_maxlength 300
```

# Test

You can evaluate your model or a pretrained model with [`test_reader.py`](test_reader.py). An example usage of the script is provided below.

```shell
python test_reader.py \
        --model_path /tmp/out \
        --eval_data eval_data.json \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name test1 \
        --checkpoint_dir out1
```
For each method we suggest in the paper, we provide the following parameters for each method:

## Early-Exit parameters

These are the parameters of the early-exit mechanism. We note that we specify the early-exit threshold ($\lambda_t$) schedule during the token generation ($t$), as follows:

```math
\lambda_t = clip_([0,1]) (\alpha \lambda + (1 - \alpha)e^{-\tau t / N} )
```

* **'--decoder_early_exit_type'** : The type of confidence estimation method to use:
                "classifier": A classifier model to estimate the confidence, given the hidden states at the current layer.
                "softmax": Applies the softmax operation to over the vocabulary prediction at the current layer. If the confidence is then the difference between the first and second highest ranked tokens.
                "state": Computes the cosine similarity between the current hidden states and the previous ones.

* **'--decoder_early_exit_thres'**: (The $\lambda$ above) The confidence threshold for the current method. If the confidence value exceeds this value, an early exit is triggered.
* **'--decoder_early_exit_tau'**: (The $\tau$ above) The tau in the confidence threshold scheduling mechanism, which modifies the rate of change.
* **'--decoder_early_exit_alpha'**: (The $\alpha$ above) The alpha in the confidence threshold scheduling mechanism, which modifies the coefficient of the scheduling component.

### Token Filtering parameters

* **'--filter'**: ether Token Filtering should be used or not.
* **'--filter_to_take_percent'**: Percentage of tokens to keep after filtering.
* **'--filter_token'**: The generated token index, where filtering needs to occur.
* **'--filter_layer'**: The decoder layer index, where filtering needs to occur.

* **'--filter_use_values'**: ether the normalization with the values tensor should be used or not.
* **'--filter_use_last_state'**: ether to use only the latest layer in the cross-attention score computation or not.

For example, a command using these parameters:

```shell
CUDA_VISIBLE_DEVICES=0 python test_reader.py \
--checkpoint_dir=/tmp/output \
--model_path=MODEL_PATH \
--name=test1 \
--per_gpu_batch_size=1 \
--answer_maxlength=300 \
--answer_minlength=50 \
--eval_data=evaluation_data.json \
--infer_bf16 \
--num_beams=4 \
--seed=42 \
--text_maxlength=512 \
--n_context=100 \
--filter_to_take_percent=0.1 \
--filter_token=3 \
--filter_layer=2 \
--filter_use_values \
--filter \
--decoder_early_exit_thres=0.8 \
--decoder_early_exit_type=classifier
```

## Setting Parameters from a File

These options override the parameter values from the CLI, by injecting the exact setting specified in an Interval Configuration file, with the following format:

```json
[
        {"n_context": 10, "filter_to_take_percent": 0.3, "filter_token": 6.0, "filter_layer": 1.0, "filter_use_values": true, "filter_use_last_state": true, "filter": true, "decoder_early_exit_thres": 0.3, "decoder_early_exit_type": "classifier", "decoder_early_exit_alpha": 0.9, "decoder_early_exit_tau": 4},
        ...
]
```

We provide the setup files for the best configurations for each dataset and each task from our paper in the [here](hyperparameter_setup_files).

* **'--setup_test_file'**: The file containing the interval configurations.
* **'--setup_test_index'**: The interval configuration to apply. For example, if the interval configuration file contains 30 entries, and the setup_test_index=4 then the fifth configuration will override all the test time arguments specified in it.



## References

[1] M. Berchansky, P. Izsak, A. Caciularu, I. Dagan, M. Wasserblat [*Optimizing Retrieval-augmented Reader Models via Token Elimination*](https://arxiv.org/abs/2310.13682)

```bibtex
@misc{berchansky2023optimizing,
      title={Optimizing Retrieval-augmented Reader Models via Token Elimination},
      author={Moshe Berchansky and Peter Izsak and Avi Caciularu and Ido Dagan and Moshe Wasserblat},
      year={2023},
      eprint={2310.13682},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

See the [LICENSE](LICENSE) file for more details.
