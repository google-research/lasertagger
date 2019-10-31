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
"""Utilities for building a LaserTagger TF model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from typing import Any, Mapping, Optional, Text
from bert import modeling
from bert import optimization
import transformer_decoder
import tensorflow as tf
from official_transformer import model_params


class LaserTaggerConfig(modeling.BertConfig):
  """Model configuration for LaserTagger."""

  def __init__(self,
               use_t2t_decoder=True,
               decoder_num_hidden_layers=1,
               decoder_hidden_size=768,
               decoder_num_attention_heads=4,
               decoder_filter_size=3072,
               use_full_attention=False,
               **kwargs):
    """Initializes an instance of LaserTagger configuration.

    This initializer expects both the BERT specific arguments and the
    Transformer decoder arguments listed below.

    Args:
      use_t2t_decoder: Whether to use the Transformer decoder (i.e.
        LaserTagger_AR). If False, the remaining args do not affect anything and
        can be set to default values.
      decoder_num_hidden_layers: Number of hidden decoder layers.
      decoder_hidden_size: Decoder hidden size.
      decoder_num_attention_heads: Number of decoder attention heads.
      decoder_filter_size: Decoder filter size.
      use_full_attention: Whether to use full encoder-decoder attention.
      **kwargs: The arguments that the modeling.BertConfig initializer expects.
    """
    super(LaserTaggerConfig, self).__init__(**kwargs)
    self.use_t2t_decoder = use_t2t_decoder
    self.decoder_num_hidden_layers = decoder_num_hidden_layers
    self.decoder_hidden_size = decoder_hidden_size
    self.decoder_num_attention_heads = decoder_num_attention_heads
    self.decoder_filter_size = decoder_filter_size
    self.use_full_attention = use_full_attention


class ModelFnBuilder(object):
  """Class for building `model_fn` closure for TPUEstimator."""

  def __init__(self, config, num_tags,
               init_checkpoint,
               learning_rate, num_train_steps,
               num_warmup_steps, use_tpu,
               use_one_hot_embeddings, max_seq_length):
    """Initializes an instance of a LaserTagger model.

    Args:
      config: LaserTagger model configuration.
      num_tags: Number of different tags to be predicted.
      init_checkpoint: Path to a pretrained BERT checkpoint (optional).
      learning_rate: Learning rate.
      num_train_steps: Number of training steps.
      num_warmup_steps: Number of warmup steps.
      use_tpu: Whether to use TPU.
      use_one_hot_embeddings: Whether to use one-hot embeddings for word
        embeddings.
      max_seq_length: Maximum sequence length.
    """
    self._config = config
    self._num_tags = num_tags
    self._init_checkpoint = init_checkpoint
    self._learning_rate = learning_rate
    self._num_train_steps = num_train_steps
    self._num_warmup_steps = num_warmup_steps
    self._use_tpu = use_tpu
    self._use_one_hot_embeddings = use_one_hot_embeddings
    self._max_seq_length = max_seq_length

  def _create_model(self, mode, input_ids, input_mask, segment_ids, labels,
                    labels_mask):
    """Creates a LaserTagger model."""
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = modeling.BertModel(
        config=self._config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=self._use_one_hot_embeddings)

    final_hidden = model.get_sequence_output()

    if self._config.use_t2t_decoder:
      # Size of the output vocabulary which contains the tags + begin and end
      # tokens used by the Transformer decoder.
      output_vocab_size = self._num_tags + 2
      params = _get_decoder_params(self._config, self._use_tpu,
                                   self._max_seq_length, output_vocab_size)
      decoder = transformer_decoder.TransformerDecoder(params, is_training)
      logits = decoder(input_mask, final_hidden, labels)
    else:
      if is_training:
        # I.e., 0.1 dropout
        final_hidden = tf.nn.dropout(final_hidden, keep_prob=0.9)

      logits = tf.layers.dense(
          final_hidden,
          self._num_tags,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
          name="output_projection")

    with tf.variable_scope("loss"):
      loss = None
      per_example_loss = None
      if mode != tf.estimator.ModeKeys.PREDICT:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        per_example_loss = tf.truediv(
            tf.reduce_sum(loss, axis=1),
            tf.dtypes.cast(tf.reduce_sum(labels_mask, axis=1), tf.float32))
        loss = tf.reduce_mean(per_example_loss)
        pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
      else:
        if self._config.use_t2t_decoder:
          pred = logits["outputs"]
          # Transformer decoder reserves the first two IDs to the begin and the
          # end token so we shift the IDs back.
          pred -= 2
        else:
          pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

      return (loss, per_example_loss, pred)

  def build(self):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
      """The `model_fn` for TPUEstimator."""

      tf.logging.info("*** Features ***")
      for name in sorted(features.keys()):
        tf.logging.info("  name = %s, shape = %s", name, features[name].shape)

      input_ids = features["input_ids"]
      input_mask = features["input_mask"]
      segment_ids = features["segment_ids"]

      labels = None
      labels_mask = None
      if mode != tf.estimator.ModeKeys.PREDICT:
        if self._config.use_t2t_decoder:
          # Account for the begin and end tokens used by Transformer.
          labels = features["labels"] + 2
        else:
          labels = features["labels"]
        labels_mask = tf.cast(features["labels_mask"], tf.float32)

      (total_loss, per_example_loss, predictions) = self._create_model(
          mode, input_ids, input_mask, segment_ids, labels, labels_mask)

      tvars = tf.trainable_variables()
      initialized_variable_names = {}
      scaffold_fn = None
      if self._init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                        self._init_checkpoint)
        if self._use_tpu:
          def tpu_scaffold():
            tf.train.init_from_checkpoint(self._init_checkpoint, assignment_map)
            return tf.train.Scaffold()

          scaffold_fn = tpu_scaffold
        else:
          tf.train.init_from_checkpoint(self._init_checkpoint, assignment_map)

      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        tf.logging.info("Initializing the model from: %s",
                        self._init_checkpoint)
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

      output_spec = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(
            total_loss, self._learning_rate, self._num_train_steps,
            self._num_warmup_steps, self._use_tpu)

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)

      elif mode == tf.estimator.ModeKeys.EVAL:
        def metric_fn(per_example_loss, labels, labels_mask, predictions):
          """Compute eval metrics."""
          accuracy = tf.cast(
              tf.reduce_all(
                  tf.logical_or(
                      tf.equal(labels, predictions),
                      ~tf.cast(labels_mask, tf.bool)),
                  axis=1), tf.float32)
          return {
              # This is equal to the Exact score if the final realization step
              # doesn't introduce errors.
              "sentence_level_acc": tf.metrics.mean(accuracy),
              "eval_loss": tf.metrics.mean(per_example_loss),
          }

        eval_metrics = (metric_fn,
                        [per_example_loss, labels, labels_mask, predictions])
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions={"pred": predictions},
            scaffold_fn=scaffold_fn)
      return output_spec

    return model_fn


def _get_decoder_params(config, use_tpu,
                        max_seq_length,
                        output_vocab_size):
  """Returns hyperparameters for TransformerDecoder.

  Args:
    config: LaserTagger model configuration.
    use_tpu: Whether to train on TPUs.
    max_seq_length: Maximum sequence length.
    output_vocab_size: Size of the output vocabulary.

  Returns:
    Hyperparameter dictionary.
  """
  params = model_params.BASE_PARAMS
  params.update(
      num_hidden_layers=config.decoder_num_hidden_layers,
      hidden_size=config.decoder_hidden_size,
      num_heads=config.decoder_num_attention_heads,
      filter_size=config.decoder_filter_size,
      vocab_size=output_vocab_size,
      use_tpu=use_tpu,
      max_length=max_seq_length,
      # This parameter should not be changed since we want the number of decoded
      # tags to equal the number of source tokens.
      extra_decode_length=0)
  return params
