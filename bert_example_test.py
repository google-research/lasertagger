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
import bert_example
import tagging_converter
import tensorflow as tf

FLAGS = flags.FLAGS


class BertExampleTest(tf.test.TestCase):

  def setUp(self):
    super(BertExampleTest, self).setUp()

    vocab_tokens = ['[CLS]', '[SEP]', '[PAD]', 'a', 'b', 'c', '##d', '##e']
    vocab_file = os.path.join(FLAGS.test_tmpdir, 'vocab.txt')
    with tf.io.gfile.GFile(vocab_file, 'w') as vocab_writer:
      vocab_writer.write(''.join([x + '\n' for x in vocab_tokens]))

    label_map = {'KEEP': 1, 'DELETE': 2}
    max_seq_length = 8
    do_lower_case = False
    converter = tagging_converter.TaggingConverter([])
    self._builder = bert_example.BertExampleBuilder(
        label_map, vocab_file, max_seq_length, do_lower_case, converter)

  def test_building_with_target(self):
    sources = ['a b ade']
    target = 'ade'
    example = self._builder.build_bert_example(sources, target)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a b a ##d ##e [SEP] [PAD]
    self.assertEqual(example.features['input_ids'], [0, 3, 4, 3, 6, 7, 1, 2])
    self.assertEqual(example.features['input_mask'], [1, 1, 1, 1, 1, 1, 1, 0])
    self.assertEqual(example.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])
    # The first two tokens are deleted (id: 1), but the third is kept (id: 2).
    self.assertEqual(example.features['labels'], [0, 2, 2, 1, 1, 1, 0, 0])
    self.assertEqual(example.features['labels_mask'], [0, 1, 1, 1, 1, 1, 0, 0])
    self.assertEqual(example.get_token_labels(), [2, 2, 1])

  def test_building_no_target_truncated(self):
    sources = ['ade bed cde']
    example = self._builder.build_bert_example(sources)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a ##d ##e b ##e ##d [SEP]
    # where the last token 'cde' has been truncated.
    self.assertEqual(example.features['input_ids'], [0, 3, 6, 7, 4, 7, 6, 1])
    self.assertEqual(example.features['input_mask'], [1, 1, 1, 1, 1, 1, 1, 1])
    self.assertEqual(example.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])

  def test_building_with_infeasible_target(self):
    sources = ['a a a a']
    target = 'c'
    example = self._builder.build_bert_example(
        sources, target, use_arbitrary_target_ids_for_infeasible_examples=True)
    # input_ids should contain the IDs for the following tokens:
    #   [CLS] a a a a [SEP] [PAD] [PAD]
    self.assertEqual(example.features['input_ids'], [0, 3, 3, 3, 3, 1, 2, 2])
    self.assertEqual(example.features['input_mask'], [1, 1, 1, 1, 1, 1, 0, 0])
    self.assertEqual(example.features['segment_ids'], [0, 0, 0, 0, 0, 0, 0, 0])
    # Labels should alternate between KEEP (1) and DELETE (2) when the target is
    # infeasible.
    self.assertEqual(example.features['labels'], [0, 1, 2, 1, 2, 0, 0, 0])
    self.assertEqual(example.features['labels_mask'], [0, 1, 1, 1, 1, 0, 0, 0])
    self.assertEqual(example.get_token_labels(), [1, 2, 1, 2])

  def test_invalid_bert_example(self):
    with self.assertRaises(ValueError):
      # The first feature list has len 2, whereas the others have len 1, so a
      # ValueError should be raised.
      bert_example.BertExample([0, 0], [0], [0], [0], [0], [0], None, 0)


if __name__ == '__main__':
  tf.test.main()
