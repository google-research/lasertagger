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
import score_lib
import tensorflow as tf

FLAGS = flags.FLAGS


class ScoreLibTest(tf.test.TestCase):

  def test_input_reading(self):
    path = os.path.join(FLAGS.test_tmpdir, 'file.txt')
    with tf.io.gfile.GFile(path, 'w') as writer:
      writer.write('source\tpred\ttarget1\ttarget2\n'
                   'Source2\tpRed2\ttarGet3\n')
    sources, predictions, target_lists = score_lib.read_data(path,
                                                             lowercase=True)
    self.assertEqual(sources, ['source', 'source2'])
    self.assertEqual(predictions, ['pred', 'pred2'])
    self.assertEqual(target_lists, [['target1', 'target2'], ['target3']])

  def test_exact_score(self):
    predictions = ['a b c', 'd ef']
    targets = [['x', 'a b c'], ['d e f']]
    exact = score_lib.compute_exact_score(predictions, targets)
    # First prediction should match but the second should not.
    self.assertEqual(exact, 0.5)

  def test_sari_score(self):
    sources = ['a b c']
    predictions = ['a <::::> b d']
    targets = [['a <::::> b c d e']]
    sari_paper, *_ = score_lib.compute_sari_scores(
        sources, predictions, targets, ignore_wikisplit_separators=True)
    sari_other, *_ = score_lib.compute_sari_scores(
        sources, predictions, targets, ignore_wikisplit_separators=False)
    self.assertLess(sari_paper, sari_other)


if __name__ == '__main__':
  tf.test.main()
