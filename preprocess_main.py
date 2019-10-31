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
"""Convert a dataset into the TFRecord format.

The resulting TFRecord file will be used when training a LaserTagger model.
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Text

from absl import app
from absl import flags
from absl import logging

import bert_example
import tagging_converter
import utils

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples to be converted to '
    'tf.Examples.')
flags.DEFINE_enum(
    'input_format', None, ['wikisplit', 'discofuse'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string('output_tfrecord', None,
                    'Path to the resulting TFRecord file.')
flags.DEFINE_string(
    'label_map_file', None,
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per '
    'line.')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.')
flags.DEFINE_bool(
    'output_arbitrary_targets_for_infeasible_examples', False,
    'Set this to True when preprocessing the development set. Determines '
    'whether to output a TF example also for sources that can not be converted '
    'to target via the available tagging operations. In these cases, the '
    'target ids will correspond to the tag sequence KEEP-DELETE-KEEP-DELETE... '
    'which should be very unlikely to be predicted by chance. This will be '
    'useful for getting more accurate eval scores during training.')


def _write_example_count(count: int) -> Text:
  """Saves the number of converted examples to a file.

  This count is used when determining the number of training steps.

  Args:
    count: The number of converted examples.

  Returns:
    The filename to which the count is saved.
  """
  count_fname = FLAGS.output_tfrecord + '.num_examples.txt'
  with tf.io.gfile.GFile(count_fname, 'w') as count_writer:
    count_writer.write(str(count))
  return count_fname


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flag_as_required('input_file')
  flags.mark_flag_as_required('input_format')
  flags.mark_flag_as_required('output_tfrecord')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('vocab_file')

  label_map = utils.read_label_map(FLAGS.label_map_file)
  converter = tagging_converter.TaggingConverter(
      tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
      FLAGS.enable_swap_tag)
  builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                            FLAGS.max_seq_length,
                                            FLAGS.do_lower_case, converter)

  num_converted = 0
  with tf.io.TFRecordWriter(FLAGS.output_tfrecord) as writer:
    for i, (sources, target) in enumerate(utils.yield_sources_and_targets(
        FLAGS.input_file, FLAGS.input_format)):
      logging.log_every_n(
          logging.INFO,
          f'{i} examples processed, {num_converted} converted to tf.Example.',
          10000)
      example = builder.build_bert_example(
          sources, target,
          FLAGS.output_arbitrary_targets_for_infeasible_examples)
      if example is None:
        continue
      writer.write(example.to_tf_example().SerializeToString())
      num_converted += 1
  logging.info(f'Done. {num_converted} examples converted to tf.Example.')
  count_fname = _write_example_count(num_converted)
  logging.info(f'Wrote:\n{FLAGS.output_tfrecord}\n{count_fname}')


if __name__ == '__main__':
  app.run(main)
