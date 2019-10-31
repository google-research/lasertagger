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

import collections
import phrase_vocabulary_optimization as vocab
import tensorflow as tf


class PhraseVocabularyOptimizationTest(tf.test.TestCase):

  def test_lcs(self):
    tokens1 = 'abcdefg'
    tokens2 = 'xzbdqwgab'
    self.assertEqual(vocab._compute_lcs(tokens1, tokens2), ['b', 'd', 'g'])

  def test_added_phrases(self):
    source = 'First sentence . Second sentence .'
    target = 'The first sentence , on the other hand , second sentence .'
    self.assertEqual(
        vocab._get_added_phrases(source, target),
        ['the', ', on the other hand ,'])

  def test_phrase_counting(self):
    data = [
        # Added phrases: ["the"]
        (['First sentence .', 'Second sentence .'],
         'The first sentence . Second sentence .'),
        # Added phrases: ["the", "and"] (requires swapping)
        (['First sentence .', 'Second sentence .'],
         'The second sentence and first sentence .'),
    ]
    counter, all_added_phrases = vocab._added_token_counts(data,
                                                           try_swapping=True)
    self.assertEqual(counter, {'the': 2, 'and': 1})
    self.assertEqual(all_added_phrases, [['the'], ['the', 'and']])

  def test_coverage_computation(self):
    all_added_phrases = [['the', 'and'], ['and'], ['he'], ['the', 'and'], []]
    phrase_counter = collections.Counter({'and': 3, 'the': 2, 'he': 1})
    matrix = vocab._construct_added_phrases_matrix(all_added_phrases,
                                                   phrase_counter)
    # Empty vocabulary should cover the last example that doesn't require any
    # added phrases.
    self.assertEqual(
        vocab._count_covered_examples(matrix, vocabulary_size=0), 1)
    # With 1 phrase ('and') in the vocabulary, we can additionally cover the
    # second example.
    self.assertEqual(
        vocab._count_covered_examples(matrix, vocabulary_size=1), 2)
    # Adding a second phrase ('the') should cover two additional examples.
    self.assertEqual(
        vocab._count_covered_examples(matrix, vocabulary_size=2), 4)
    self.assertEqual(
        vocab._count_covered_examples(matrix, vocabulary_size=3), 5)
    self.assertEqual(
        vocab._count_covered_examples(matrix, vocabulary_size=10), 5)


if __name__ == '__main__':
  tf.test.main()
