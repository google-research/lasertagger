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

# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tagging
import tagging_converter
import tensorflow as tf


def tags_to_str(tags):
  if not tags:
    return ''
  return '--'.join(map(str, tags))


class TaggingConverterTest(parameterized.TestCase):

  @parameterized.parameters(
      # A simple test.
      {
          'input_texts': ['Turing was born in 1912 .', 'Turing died in 1954 .'],
          'target': 'Turing was born in 1912 and died in 1954 .',
          'phrase_vocabulary': ['and'],
          'target_tags': [
              'KEEP', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'DELETE|and', 'DELETE',
              'KEEP', 'KEEP', 'KEEP', 'KEEP'
          ],
      },
      # Test special characters.
      {
          'input_texts': ['Turing was born in 1912 .', 'Turing died in 1954 .'],
          'target': 'Turing was born in 1912 ädåö died in 1954 .',
          'phrase_vocabulary': ['ädåö'],
          'target_tags': [
              'KEEP', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'DELETE|ädåö', 'DELETE',
              'KEEP', 'KEEP', 'KEEP', 'KEEP'
          ],
      },
      # Test swapping.
      {
          'input_texts': ['Turing was born in 1912 .', 'Turing died in 1954 .'],
          'target': 'Turing died in 1954 and was born in 1912 .',
          'phrase_vocabulary': ['and'],
          'target_tags': [
              'DELETE', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'SWAP', 'KEEP', 'KEEP',
              'KEEP', 'KEEP', 'DELETE|and'
          ],
      },
      # Test complex swapping.
      {
          'input_texts': ['Turing was born in 1912 .',
                          'Turing was a pioneer in TCS .'],
          'target': 'Turing , a pioneer in TCS , was born in 1912 .',
          'phrase_vocabulary': [','],
          'target_tags': [
              'DELETE', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'SWAP', 'KEEP',
              'DELETE|,', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'DELETE|,'
          ],
      },
      # Test that unnecessary phrases are not added.
      {
          'input_texts': ['A . And B .'],
          'target': 'A , and B .',
          'phrase_vocabulary': [',', 'and', ', and'],
          # Although, it would be possible to add ", and" and delete "And", this
          # shouldn't happen so that the tag sequences are as simple as
          # possible.
          'target_tags': ['KEEP', 'DELETE|,', 'KEEP', 'KEEP', 'KEEP'],
      },
      # Test that necessary phrases are added.
      {
          'input_texts': ['A . And B .'],
          'target': 'A , and B .',
          'phrase_vocabulary': [', and'],
          # Now we need to delete "And" since "," is not in the vocabulary
          # anymore.
          'target_tags': ['KEEP', 'DELETE|, and', 'DELETE', 'KEEP', 'KEEP'],
      },
  )
  def test_matching_conversion(self, input_texts, target, phrase_vocabulary,
                               target_tags):
    task = tagging.EditingTask(input_texts)
    converter = tagging_converter.TaggingConverter(phrase_vocabulary)
    tags = converter.compute_tags(task, target)
    self.assertEqual(tags_to_str(tags), tags_to_str(target_tags))

  def test_no_match(self):
    input_texts = ['Turing was born in 1912 .', 'Turing died in 1954 .']
    target = 'Turing was born in 1912 and died in 1954 .'
    task = tagging.EditingTask(input_texts)
    phrase_vocabulary = ['but']
    converter = tagging_converter.TaggingConverter(phrase_vocabulary)
    tags = converter.compute_tags(task, target)
    # Vocabulary doesn't contain "and" so the inputs can't be converted to the
    # target.
    self.assertFalse(tags)

  def test_first_deletion_idx_computation(self):
    converter = tagging_converter.TaggingConverter([])
    tag_strs = ['KEEP', 'DELETE', 'DELETE', 'KEEP']
    tags = [tagging.Tag(s) for s in tag_strs]
    source_token_idx = 3
    idx = converter._find_first_deletion_idx(source_token_idx, tags)
    self.assertEqual(idx, 1)

  def test_phrase_vocabulary_extraction(self):
    label_map = {'KEEP|, and': 0, 'DELETE|but': 1, 'DELETE': 2, 'KEEP|and': 3,
                 'DELETE|and': 4}
    self.assertEqual(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
        {', and', 'but', 'and'})


if __name__ == '__main__':
  tf.test.main()
