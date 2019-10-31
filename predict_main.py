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
"""Compute realized predictions for a dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import bert_example
import predict_utils
import tagging_converter
import utils

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples for which to compute '
    'predictions.')
flags.DEFINE_enum(
    'input_format', None, ['wikisplit', 'discofuse'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string(
    'output_file', None,
    'Path to the TSV file where the predictions are written to.')
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
flags.DEFINE_string('saved_model', None, 'Path to an exported TF model.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flag_as_required('input_file')
  flags.mark_flag_as_required('input_format')
  flags.mark_flag_as_required('output_file')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('vocab_file')
  flags.mark_flag_as_required('saved_model')

  label_map = utils.read_label_map(FLAGS.label_map_file)
  converter = tagging_converter.TaggingConverter(
      tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
      FLAGS.enable_swap_tag)
  builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                            FLAGS.max_seq_length,
                                            FLAGS.do_lower_case, converter)
  predictor = predict_utils.LaserTaggerPredictor(
      tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
      label_map)

  num_predicted = 0
  with tf.gfile.Open(FLAGS.output_file, 'w') as writer:
    for i, (sources, target) in enumerate(utils.yield_sources_and_targets(
        FLAGS.input_file, FLAGS.input_format)):
      logging.log_every_n(
          logging.INFO,
          f'{i} examples processed, {num_predicted} converted to tf.Example.',
          100)
      prediction = predictor.predict(sources)
      writer.write(f'{" ".join(sources)}\t{prediction}\t{target}\n')
      num_predicted += 1
  logging.info(f'{num_predicted} predictions saved to:\n{FLAGS.output_file}')


if __name__ == '__main__':
  app.run(main)
