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

import os

from absl import flags
from absl.testing import parameterized

import bert_example
import predict_utils
import tagging_converter

import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


class DummyPredictor(object):

  def __init__(self, output_ids):
    """Initializes for a dummy predictor that always predicts `output_ids`."""
    self._output_ids = output_ids

  def __call__(self, example=None):
    # Prepend and append IDs for the begin and the end token.
    return {'pred': np.array([[0] + self._output_ids + [0]])}


class PredictUtilsTest(parameterized.TestCase):

  def setUp(self):
    super(PredictUtilsTest, self).setUp()

    vocab_tokens = ['[CLS]', '[SEP]', '[PAD]', 'a', 'b', 'c', '##d', '##e']
    vocab_file = os.path.join(FLAGS.test_tmpdir, 'vocab.txt')
    with tf.io.gfile.GFile(vocab_file, 'w') as vocab_writer:
      vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))

    self._label_map = {'KEEP': 0, 'DELETE': 1, 'KEEP|and': 2}
    max_seq_length = 8
    do_lower_case = False
    converter = tagging_converter.TaggingConverter([])
    self._builder = bert_example.BertExampleBuilder(
        self._label_map, vocab_file, max_seq_length, do_lower_case, converter)

  @parameterized.parameters(
      {
          'prediction_ids': [1, 0, 2],
          'target': 'b and ade',
      },
      # When predictions are truncated, the missing tags should be set to KEEP.
      {
          'prediction_ids': [1, 0],
          'target': 'b ade',
      },
  )
  def test_predictions(self, prediction_ids, target):
    sources = ['a b ade']
    predictor = predict_utils.LaserTaggerPredictor(
        DummyPredictor(prediction_ids), self._builder, self._label_map)
    realized_prediction = predictor.predict(sources)
    self.assertEqual(realized_prediction, target)


if __name__ == '__main__':
  tf.test.main()
