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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import run_lasertagger_utils

import tensorflow as tf


def _get_model_builder(use_t2t_decoder=True):
  """Returns a LaserTagger model_fn builder."""
  config_json = {
      "hidden_size": 4,
      "intermediate_size": 8,
      "max_position_embeddings": 8,
      "num_attention_heads": 1,
      "num_hidden_layers": 1,
      "vocab_size": 8,
      "use_t2t_decoder": use_t2t_decoder,
      "decoder_num_hidden_layers": 1,
      "decoder_hidden_size": 4,
      "decoder_num_attention_heads": 1,
      "decoder_filter_size": 4,
      "use_full_attention": False,
  }
  config = run_lasertagger_utils.LaserTaggerConfig(**config_json)
  return run_lasertagger_utils.ModelFnBuilder(
      config=config,
      num_tags=2,
      init_checkpoint=None,
      learning_rate=1e-4,
      num_train_steps=10,
      num_warmup_steps=1,
      use_tpu=False,
      use_one_hot_embeddings=False,
      max_seq_length=128)


class RunLasertaggerUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RunLasertaggerUtilsTest, self).setUp()
    self._features = {
        "input_ids": [[0, 2, 3, 1, 0]],
        "input_mask": [[1, 1, 1, 1, 0]],
        "segment_ids": [[0, 0, 0, 0, 0]],
        "labels": [[0, 0, 1, 0, 0]],
        "labels_mask": [[0, 1, 1, 0, 0]],
    }
    self._features = {k: tf.convert_to_tensor(v)
                      for (k, v) in self._features.items()}

  @parameterized.parameters(True, False)
  def test_create_model(self, use_t2t_decoder):
    """Test creating LaserTagger_AR model."""
    input_ids = tf.constant([[0, 2, 3, 1, 0]], dtype=tf.int64)
    input_mask = tf.constant([[1, 1, 1, 1, 0]], dtype=tf.int64)
    segment_ids = tf.constant([[0, 0, 0, 0, 0]], dtype=tf.int64)
    labels = tf.constant([[0, 0, 1, 0, 0]], dtype=tf.int64)
    labels_mask = tf.constant([[0, 1, 1, 0, 0]], dtype=tf.int64)

    model_fn_builder = _get_model_builder(use_t2t_decoder)
    (loss, _, pred) = model_fn_builder._create_model(
        tf.estimator.ModeKeys.TRAIN, input_ids, input_mask, segment_ids, labels,
        labels_mask)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out = sess.run({"loss": loss, "pred": pred})

      self.assertEqual(out["loss"].shape, ())
      self.assertEqual(out["pred"].shape, labels.shape)

  def test_model_fn_train(self):
    with self.session() as sess:
      model_fn_builder = _get_model_builder()
      model_fn = model_fn_builder.build()
      output_spec = model_fn(
          self._features,
          labels=None,
          mode=tf.estimator.ModeKeys.TRAIN,
          params=None)

      sess.run([tf.global_variables_initializer(),
                tf.local_variables_initializer()])
      loss = sess.run(output_spec.loss)
      self.assertAllEqual(loss.shape, [])

  def test_model_fn_eval(self):
    with self.session() as sess:
      model_fn_builder = _get_model_builder()
      model_fn = model_fn_builder.build()
      output_spec = model_fn(
          self._features,
          labels=None,
          mode=tf.estimator.ModeKeys.EVAL,
          params=None)
      metric_fn = output_spec.eval_metrics[0]
      metric_fn_args = output_spec.eval_metrics[1]
      self.assertLen(metric_fn_args, 4)
      metrics = metric_fn(*metric_fn_args)
      sess.run([tf.global_variables_initializer(),
                tf.local_variables_initializer()])

      def check_metric_shape(metric):
        val_node, update_op = metric
        sess.run(update_op)
        val = sess.run(val_node)
        self.assertAllEqual(val.shape, [])

      self.assertLen(metrics, 2)
      check_metric_shape(metrics["eval_loss"])
      check_metric_shape(metrics["sentence_level_acc"])


if __name__ == "__main__":
  tf.test.main()
