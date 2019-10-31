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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tagging
import tensorflow as tf


class TaggingTest(tf.test.TestCase):

  def test_tag_parsing(self):
    tag = tagging.Tag('KEEP')
    self.assertEqual(tag.tag_type, tagging.TagType.KEEP)
    self.assertEqual(tag.added_phrase, '')

    tag = tagging.Tag('DELETE|, and she')
    self.assertEqual(tag.tag_type, tagging.TagType.DELETE)
    self.assertEqual(tag.added_phrase, ', and she')

    tag = tagging.Tag('SWAP|asdf | foo')
    self.assertEqual(tag.tag_type, tagging.TagType.SWAP)
    self.assertEqual(tag.added_phrase, 'asdf | foo')

    with self.assertRaises(ValueError):
      tagging.Tag('NON_EXISTING_TAG')

  def test_copying(self):
    input_texts = ['Turing was born in 1912 in London .']
    tag_strs = ['KEEP'] * 8
    tags = [tagging.Tag(s) for s in tag_strs]
    task = tagging.EditingTask(input_texts)
    self.assertEqual(task.realize_output(tags), input_texts[0])

    # With multiple inputs.
    input_texts = ['a B', 'c D e', 'f g']
    tag_strs = ['KEEP'] * 7
    tags = [tagging.Tag(s) for s in tag_strs]
    task = tagging.EditingTask(input_texts)
    self.assertEqual(task.realize_output(tags), 'a B c D e f g')

  def test_deletion(self):
    input_texts = ['Turing was born in 1912 in London .']
    tag_strs = [
        'KEEP', 'DELETE', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'DELETE'
    ]
    tags = [tagging.Tag(s) for s in tag_strs]
    task = tagging.EditingTask(input_texts)
    # "was" and "." should have been removed.
    self.assertEqual(task.realize_output(tags), 'Turing born in 1912 in London')

  def test_phrase_adding(self):
    input_texts = ['Turing was born in 1912 in London .']
    tag_strs = [
        'KEEP', 'DELETE|, a pioneer in TCS ,', 'KEEP', 'KEEP', 'KEEP', 'KEEP',
        'KEEP', 'KEEP'
    ]
    tags = [tagging.Tag(s) for s in tag_strs]
    task = tagging.EditingTask(input_texts)
    self.assertEqual(
        task.realize_output(tags),
        'Turing , a pioneer in TCS , born in 1912 in London .')

  def test_swapping(self):
    input_texts = [
        'Turing was born in 1912 in London .', 'Turing died in 1954 .'
    ]
    tag_strs = [
        'KEEP', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'SWAP', 'KEEP',
        'KEEP', 'KEEP', 'KEEP', 'KEEP'
    ]
    tags = [tagging.Tag(s) for s in tag_strs]
    task = tagging.EditingTask(input_texts)
    self.assertEqual(
        task.realize_output(tags),
        'Turing died in 1954 . Turing was born in 1912 in London .')

  def test_invalid_swapping(self):
    # When SWAP tag is assigned to other than the last token of the first of two
    # sentences, it should be treated as KEEP.
    input_texts = [
        'Turing was born in 1912 in London .', 'Turing died in 1954 .'
    ]
    tag_strs = [
        'KEEP', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'KEEP', 'SWAP', 'KEEP', 'KEEP',
        'KEEP', 'KEEP', 'KEEP', 'KEEP'
    ]
    tags = [tagging.Tag(s) for s in tag_strs]
    task = tagging.EditingTask(input_texts)
    self.assertEqual(
        task.realize_output(tags),
        'Turing was born in 1912 in London . Turing died in 1954 .')

  def test_swapping_complex(self):
    input_texts = ['Dylan won Nobel prize .', 'Dylan is an American musician .']
    tag_strs = [
        'DELETE', 'KEEP', 'KEEP', 'KEEP', 'SWAP', 'KEEP', 'DELETE|,', 'KEEP',
        'KEEP', 'KEEP', 'DELETE|,'
    ]
    tags = [tagging.Tag(s) for s in tag_strs]
    task = tagging.EditingTask(input_texts)
    self.assertEqual(
        task.realize_output(tags),
        'Dylan , an American musician , won Nobel prize .')

  def test_casing(self):
    input_texts = ['A b .', 'Cc dd .']
    # Test lowcasing after a period has been removed.
    tag_strs = ['KEEP', 'KEEP', 'DELETE', 'KEEP', 'KEEP', 'KEEP']
    tags = [tagging.Tag(s) for s in tag_strs]
    task = tagging.EditingTask(input_texts)
    self.assertEqual(task.realize_output(tags), 'A b cc dd .')

    # Test upcasing after the first upcased token has been removed.
    tag_strs = ['KEEP', 'KEEP', 'KEEP', 'DELETE', 'KEEP', 'KEEP']
    tags = [tagging.Tag(s) for s in tag_strs]
    task = tagging.EditingTask(input_texts)
    self.assertEqual(task.realize_output(tags), 'A b . Dd .')

  def test_wrong_number_of_tags(self):
    input_texts = ['1 2']
    tags = [tagging.Tag('KEEP')]
    task = tagging.EditingTask(input_texts)
    with self.assertRaises(ValueError):
      task.realize_output(tags)


if __name__ == '__main__':
  tf.test.main()
