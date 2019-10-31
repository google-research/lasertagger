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
"""Utility functions for LaserTagger."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
from typing import Iterator, Mapping, Sequence, Text, Tuple

import tensorflow as tf


def get_token_list(text):
  """Returns a list of tokens.

  This function expects that the tokens in the text are separated by space
  character(s). Example: "ca n't , touch". This is the case at least for the
  public DiscoFuse and WikiSplit datasets.

  Args:
    text: String to be split into tokens.
  """
  return text.split()


def yield_sources_and_targets(
    input_file,
    input_format):
  """Reads and yields source lists and targets from the input file.

  Args:
    input_file: Path to the input file.
    input_format: Format of the input file.

  Yields:
    Tuple with (list of source texts, target text).
  """
  if input_format == 'wikisplit':
    yield_example_fn = _yield_wikisplit_examples
  elif input_format == 'discofuse':
    yield_example_fn = _yield_discofuse_examples
  else:
    raise ValueError('Unsupported input_format: {}'.format(input_format))

  for sources, target in yield_example_fn(input_file):
    yield sources, target


def _yield_wikisplit_examples(
    input_file):
  # The Wikisplit format expects a TSV file with the source on the first and the
  # target on the second column.
  with tf.io.gfile.GFile(input_file) as f:
    for line in f:
      source, target = line.rstrip('\n').split('\t')
      yield [source], target


def _yield_discofuse_examples(
    input_file):
  """Yields DiscoFuse examples.

  The documentation for this format:
  https://github.com/google-research-datasets/discofuse#data-format

  Args:
    input_file: Path to the input file.
  """
  with tf.io.gfile.GFile(input_file) as f:
    for i, line in enumerate(f):
      if i == 0:  # Skip the header line.
        continue
      coherent_1, coherent_2, incoherent_1, incoherent_2, _, _, _, _ = (
          line.rstrip('\n').split('\t'))
      # Strip because the second coherent sentence might be empty.
      fusion = (coherent_1 + ' ' + coherent_2).strip()
      yield [incoherent_1, incoherent_2], fusion


def read_label_map(path):
  """Returns label map read from the given path."""
  with tf.io.gfile.GFile(path) as f:
    if path.endswith('.json'):
      return json.load(f)
    else:
      label_map = {}
      empty_line_encountered = False
      for tag in f:
        tag = tag.strip()
        if tag:
          label_map[tag] = len(label_map)
        else:
          if empty_line_encountered:
            raise ValueError(
                'There should be no empty lines in the middle of the label map '
                'file.'
            )
          empty_line_encountered = True
      return label_map
