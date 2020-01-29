# LaserTagger

LaserTagger is a text-editing model which predicts a sequence of token-level
edit operations to transform a source text into a target text. The model
currently supports four different edit operations:

1. *Keep* the token.
2. *Delete* the token.
3. *Add* a phrase before the token.
4. *Swap* the order of input sentences (if there are two of them).

Operation 3 can be combined with 1 and 2. Compared to sequence-to-sequence
models, LaserTagger is (1) less prone to hallucination, (2) more data efficient,
and (3) faster at inference time.

A detailed method description and evaluation can be found in our EMNLP'19 paper:
[https://arxiv.org/abs/1909.01187](https://arxiv.org/abs/1909.01187)

LaserTagger is built on Python 3, Tensorflow and
[BERT](https://github.com/google-research/bert). It works with CPU, GPU, and
Cloud TPU.

## Usage Instructions

Running an experiment with LaserTagger consists of the following steps:

1. Optimize the vocabulary of phrases that can be added by LaserTagger.
2. Convert target texts into target tag sequences.
3. Finetune a pretrained BERT model to predict the tags.
4. Compute predictions.
5. Evaluate the predictions.

Next we go through these steps, using the Split-and-Rephrase
([WikiSplit](https://github.com/google-research-datasets/wiki-split)) task as a
running example.

You can run all of the steps with

```
sh run_wikisplit_experiment.sh
```

after setting the paths in the beginning of the script.

**Note:** Text should be tokenized with spaces separating the tokens before applying LaserTagger.

### 1. Phrase Vocabulary Optimization

Download the [WikiSplit](https://github.com/google-research-datasets/wiki-split)
dataset and run the following command to find a set of phrases that the model is
allowed to add.

```
export WIKISPLIT_DIR=/path/to/wikisplit
export OUTPUT_DIR=/path/to/output

python phrase_vocabulary_optimization.py \
  --input_file=${WIKISPLIT_DIR}/train.tsv \
  --input_format=wikisplit \
  --vocabulary_size=500 \
  --max_input_examples=1000000 \
  --output_file=${OUTPUT_DIR}/label_map.txt
```

Note that you can also set `max_input_examples` to a smaller value to get a
reasonable vocabulary, but then you should sort the dataset rows in the case of
WikiSplit. The rows are in an alphabetical order so taking first *k* of them
might not give you a representative sample of the data.

### 2. Converting Target Texts to Tags

Download a pretrained BERT model from the
[official repository](https://github.com/google-research/bert#pre-trained-models).
We've used the 12-layer ''BERT-Base, Cased'' model for all of our experiments.
Then convert the original TSV datasets into TFRecord format.

```
export BERT_BASE_DIR=/path/to/cased_L-12_H-768_A-12

python preprocess_main.py \
  --input_file=${WIKISPLIT_DIR}/tune.tsv \
  --input_format=wikisplit \
  --output_tfrecord=${OUTPUT_DIR}/tune.tf_record \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --output_arbitrary_targets_for_infeasible_examples=true

python preprocess_main.py \
    --input_file=${WIKISPLIT_DIR}/train.tsv \
    --input_format=wikisplit \
    --output_tfrecord=${OUTPUT_DIR}/train.tf_record \
    --label_map_file=${OUTPUT_DIR}/label_map.txt \
    --vocab_file=${BERT_BASE_DIR}/vocab.txt \
    --output_arbitrary_targets_for_infeasible_examples=false
```

### 3. Model Training

Model hyperparameters are specified in [lasertagger_config.json](configs/lasertagger_config.json). This configuration file extends
`bert_config.json` which comes with the zipped pretrained BERT model.

Note that if you want to **switch
from using LaserTagger_FF to LaserTagger_AR**, you should set
`"use_t2t_decoder": true` in the LaserTagger config. The latter is usually more
accurate, whereas the former runs inference faster.

Train the model on CPU/GPU.

```
# Check these numbers from the "*.num_examples" files created in step 2.
export NUM_TRAIN_EXAMPLES=310922
export NUM_EVAL_EXAMPLES=5000
export CONFIG_FILE=configs/lasertagger_config.json
export EXPERIMENT=wikisplit_experiment_name

python run_lasertagger.py \
  --training_file=${OUTPUT_DIR}/train.tf_record \
  --eval_file=${OUTPUT_DIR}/tune.tf_record \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --model_config_file=${CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
  --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
  --do_train=true \
  --do_eval=true \
  --train_batch_size=256 \
  --save_checkpoints_steps=500 \
  --num_train_examples=${NUM_TRAIN_EXAMPLES} \
  --num_eval_examples=${NUM_EVAL_EXAMPLES}
```

To train on Cloud TPU, you should additionally set:

```
  --use_tpu=true \
  --tpu_name=${TPU_NAME}
```

Please see [BERT TPU instructions](https://github.com/google-research/bert#fine-tuning-with-cloud-tpus) and the
[Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist)
for how to use Cloud TPUs.

### 4. Prediction

First you need to export your model.

```
python run_lasertagger.py \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --model_config_file=${CONFIG_FILE} \
  --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
  --do_export=true \
  --export_path=${OUTPUT_DIR}/models/${EXPERIMENT}/export
```

You can additionally use `init_checkpoint` to specify which checkpoint to export
(the default is to export the latest).

Compute the predicted tags and realize the output text with:

```
export SAVED_MODEL_DIR=/path/to/exported/model
export PREDICTION_FILE=${OUTPUT_DIR}/models/${EXPERIMENT}/pred.tsv

python predict_main.py \
  --input_file=${WIKISPLIT_DIR}/validation.tsv \
  --input_format=wikisplit \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --saved_model=${SAVED_MODEL_DIR}
```

Note that the above will run inference with batch size of 1 so it's not optimal
in terms of inference time.

### 5. Evaluation

Compute the evaluation scores.

```
python score_main.py --prediction_file=${PREDICTION_FILE}
```

Example output:

```
Exact score:     15.220
SARI score:      61.668
 KEEP score:     93.059
 ADDITION score: 32.168
 DELETION score: 59.778
```

## How to Cite LaserTagger

```
@inproceedings{malmi2019lasertagger,
  title={Encode, Tag, Realize: High-Precision Text Editing},
  author={Eric Malmi and Sebastian Krause and Sascha Rothe and Daniil Mirylenka and Aliaksei Severyn},
  booktitle={EMNLP-IJCNLP},
  year={2019}
}
```

## License

Apache 2.0; see [LICENSE](LICENSE) for details.

## Disclaimer

This repository contains a Python reimplementation of our original
C++ code used for the paper and thus some discrepancies compared to the paper
results are possible. However, we've verified that we get the similar results on
the WikiSplit dataset.

This is not an official Google product.
