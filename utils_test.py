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

import json
import os

from absl import flags
import utils
import tensorflow as tf


FLAGS = flags.FLAGS


class UtilsTest(tf.test.TestCase):

  def test_read_wikisplit(self):
    path = os.path.join(FLAGS.test_tmpdir, "file.txt")
    with tf.io.gfile.GFile(path, "w") as writer:
      writer.write("Source sentence .\tTarget sentence .\n")
      writer.write("2nd source .\t2nd target .")
    examples = list(utils.yield_sources_and_targets(path, "wikisplit"))
    self.assertEqual(examples, [(["Source sentence ."], "Target sentence ."),
                                (["2nd source ."], "2nd target .")])

  def test_read_discofuse(self):
    path = os.path.join(FLAGS.test_tmpdir, "file.txt")
    with tf.io.gfile.GFile(path, "w") as writer:
      writer.write(
          "coherent_first_sentence\tcoherent_second_sentence\t"
          "incoherent_first_sentence\tincoherent_second_sentence\t"
          "discourse_type\tconnective_string\thas_coref_type_pronoun\t"
          "has_coref_type_nominal\n"
      )
      writer.write(
          "1st sentence .\t2nd sentence .\t1st inc sent .\t2nd inc sent .\t"
          "PAIR_ANAPHORA\t\t1.0\t0.0\n"
      )
      writer.write(
          "1st sentence and 2nd sentence .\t\t1st inc sent .\t"
          "2nd inc sent .\tSINGLE_S_COORD_ANAPHORA\tand\t1.0\t0.0"
      )
    examples = list(utils.yield_sources_and_targets(path, "discofuse"))
    self.assertEqual(examples, [(["1st inc sent .", "2nd inc sent ."],
                                 "1st sentence . 2nd sentence ."),
                                (["1st inc sent .", "2nd inc sent ."],
                                 "1st sentence and 2nd sentence .")])

  def test_read_label_map(self):
    orig_label_map = {"KEEP": 0, "DELETE": 1}
    path = os.path.join(FLAGS.test_tmpdir, "file.json")
    with tf.io.gfile.GFile(path, "w") as writer:
      json.dump(orig_label_map, writer)
    label_map = utils.read_label_map(path)
    self.assertEqual(label_map, orig_label_map)

  def test_read_non_json_label_map(self):
    path = os.path.join(FLAGS.test_tmpdir, "file.txt")
    with tf.io.gfile.GFile(path, "w") as writer:
      writer.write("KEEP\nDELETE\n\n")
    label_map = utils.read_label_map(path)
    self.assertEqual(label_map, {"KEEP": 0, "DELETE": 1})


if __name__ == "__main__":
  tf.test.main()
