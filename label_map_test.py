# coding=utf-8
# Copyright 2026 The Google Research Authors.
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
"""Regression tests for blank-line validation in text label maps."""

import json

import pytest
import tagging
import tensorflow as tf
import utils


@pytest.mark.parametrize('newline', ['\n', '\r\n'])
@pytest.mark.parametrize('lines', [
    ['KEEP', '', 'DELETE'],
    ['KEEP', '   ', 'DELETE'],
    ['KEEP', '', '', 'DELETE'],
    ['', 'KEEP'],
])
def test_reject_blank_lines_before_later_tags(tmp_path, lines, newline):
  path = tmp_path / 'labels.txt'
  path.write_bytes(newline.join(lines).encode('utf-8'))
  with pytest.raises(ValueError, match='no empty lines in the middle'):
    utils.read_label_map(str(path))


@pytest.mark.parametrize('suffix', ['', '\n', '\n\n', '\n\n\n', '\n \n\t\n'])
def test_allow_only_trailing_blank_lines(tmp_path, suffix):
  path = tmp_path / 'labels.txt'
  path.write_text('KEEP\nDELETE\nKEEP|and' + suffix, encoding='utf-8')
  assert utils.read_label_map(str(path)) == {
      'KEEP': 0,
      'DELETE': 1,
      'KEEP|and': 2,
  }


@pytest.mark.parametrize('contents', ['', '\n', '\n\n', ' \n\t\n'])
def test_blank_only_label_maps_have_no_labels(tmp_path, contents):
  path = tmp_path / 'labels.txt'
  path.write_text(contents, encoding='utf-8')
  assert utils.read_label_map(str(path)) == {}


def test_preserve_tag_whitespace_unicode_and_order(tmp_path):
  path = tmp_path / 'labels.txt'
  path.write_text('  KEEP \nDELETE\n KEEP|déjà vu \n\n', encoding='utf-8')
  labels = utils.read_label_map(str(path))
  assert list(labels.items()) == [('KEEP', 0), ('DELETE', 1),
                                  ('KEEP|déjà vu', 2)]
  task = tagging.EditingTask(['one two'])
  by_id = {index: tagging.Tag(tag) for tag, index in labels.items()}
  assert task.realize_output([by_id[2], by_id[1]]) == 'déjà vu one'


def test_json_label_map_is_unchanged(tmp_path):
  path = tmp_path / 'labels.json'
  expected = {'KEEP': 5, 'DELETE': 3, 'KEEP|é': 8}
  path.write_text(json.dumps(expected, indent=2) + '\n\n', encoding='utf-8')
  assert utils.read_label_map(str(path)) == expected


def test_malformed_json_preserves_error(tmp_path):
  path = tmp_path / 'labels.json'
  path.write_text('{invalid', encoding='utf-8')
  with pytest.raises(json.JSONDecodeError):
    utils.read_label_map(str(path))


def test_missing_file_preserves_error(tmp_path):
  with pytest.raises(tf.errors.NotFoundError):
    utils.read_label_map(str(tmp_path / 'missing.txt'))
