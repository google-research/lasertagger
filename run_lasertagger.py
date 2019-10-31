# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""BERT-based LaserTagger runner."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Text
from absl import flags

import run_lasertagger_utils
import utils

import tensorflow as tf

FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string("training_file", None,
                    "Path to the TFRecord training file.")
flags.DEFINE_string("eval_file", None, "Path to the the TFRecord dev file.")
flags.DEFINE_string(
    "label_map_file", None,
    "Path to the label map file. Either a JSON file ending with '.json', that "
    "maps each possible tag to an ID, or a text file that has one tag per "
    "line.")
flags.DEFINE_string(
    "model_config_file", None,
    "The config json file specifying the model architecture.")
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written. If "
    "`init_checkpoint' is not provided when exporting, the latest checkpoint "
    "from this directory will be exported.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint, usually from a pre-trained BERT model. In the case of "
    "exporting, one can optionally provide path to a particular checkpoint to "
    "be exported here.")
flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter than "
    "this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
flags.DEFINE_bool("do_export", False, "Whether to export a trained model.")
flags.DEFINE_bool("eval_all_checkpoints", False, "Run through all checkpoints.")
flags.DEFINE_integer(
    "eval_timeout", 600,
    "The maximum amount of time (in seconds) for eval worker to wait between "
    "checkpoints.")


flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")
flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")
flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")
flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")
flags.DEFINE_integer(
    "num_train_examples", None,
    "Number of training examples. This is used to determine the number of "
    "training steps to respect the `num_train_epochs` flag.")
flags.DEFINE_integer(
    "num_eval_examples", None,
    "Number of eval examples. This is used to determine the number of "
    "eval steps to go through the eval file once.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
flags.DEFINE_string("master", None,
                    "Optional address of the master for the workers.")
flags.DEFINE_string("export_path", None, "Path to save the exported model.")


def file_based_input_fn_builder(input_file, max_seq_length,
                                is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
      "labels": tf.FixedLenFeature([max_seq_length], tf.int64),
      "labels_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    d = tf.data.TFRecordDataset(input_file)
    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=params["batch_size"],
            drop_remainder=drop_remainder))
    return d

  return input_fn


def _calculate_steps(num_examples, batch_size, num_epochs, warmup_proportion=0):
  """Calculates the number of steps.

  Args:
    num_examples: Number of examples in the dataset.
    batch_size: Batch size.
    num_epochs: How many times we should go through the dataset.
    warmup_proportion: Proportion of warmup steps.

  Returns:
    Tuple (number of steps, number of warmup steps).
  """
  steps = int(num_examples / batch_size * num_epochs)
  warmup_steps = int(warmup_proportion * steps)
  return steps, warmup_steps


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not (FLAGS.do_train or FLAGS.do_eval or FLAGS.do_export):
    raise ValueError("At least one of `do_train`, `do_eval` or `do_export` must"
                     " be True.")

  model_config = run_lasertagger_utils.LaserTaggerConfig.from_json_file(
      FLAGS.model_config_file)

  if FLAGS.max_seq_length > model_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, model_config.max_position_embeddings))

  if not FLAGS.do_export:
    tf.io.gfile.makedirs(FLAGS.output_dir)

  num_tags = len(utils.read_label_map(FLAGS.label_map_file))

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=20,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          per_host_input_for_training=is_per_host,
          eval_training_input_configuration=tf.contrib.tpu.InputPipelineConfig
          .SLICED))

  if FLAGS.do_train:
    num_train_steps, num_warmup_steps = _calculate_steps(
        FLAGS.num_train_examples, FLAGS.train_batch_size,
        FLAGS.num_train_epochs, FLAGS.warmup_proportion)
  else:
    num_train_steps, num_warmup_steps = None, None

  model_fn = run_lasertagger_utils.ModelFnBuilder(
      config=model_config,
      num_tags=num_tags,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      max_seq_length=FLAGS.max_seq_length).build()

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_input_fn = file_based_input_fn_builder(
        input_file=FLAGS.training_file,
        max_seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      # Eval will be slightly WRONG on the TPU because it will truncate
      # the last batch.
      eval_steps, _ = _calculate_steps(FLAGS.num_eval_examples,
                                       FLAGS.eval_batch_size, 1)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=FLAGS.eval_file,
        max_seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    for ckpt in tf.contrib.training.checkpoints_iterator(
        FLAGS.output_dir, timeout=FLAGS.eval_timeout):
      result = estimator.evaluate(input_fn=eval_input_fn, checkpoint_path=ckpt,
                                  steps=eval_steps)
      for key in sorted(result):
        tf.logging.info("  %s = %s", key, str(result[key]))

  if FLAGS.do_export:
    tf.logging.info("Exporting the model...")
    def serving_input_fn():
      def _input_fn():
        features = {
            "input_ids": tf.placeholder(tf.int64, [None, None]),
            "input_mask": tf.placeholder(tf.int64, [None, None]),
            "segment_ids": tf.placeholder(tf.int64, [None, None]),
        }
        return tf.estimator.export.ServingInputReceiver(
            features=features, receiver_tensors=features)
      return _input_fn

    estimator.export_saved_model(
        FLAGS.export_path,
        serving_input_fn(),
        checkpoint_path=FLAGS.init_checkpoint)


if __name__ == "__main__":
  flags.mark_flag_as_required("model_config_file")
  flags.mark_flag_as_required("label_map_file")
  tf.app.run()
