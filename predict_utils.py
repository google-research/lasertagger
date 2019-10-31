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
"""Utility functions for running inference with a LaserTagger model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Mapping, Sequence, Text

import bert_example
import tagging


class LaserTaggerPredictor(object):
  """Class for computing and realizing predictions with LaserTagger."""

  def __init__(self, tf_predictor,
               example_builder,
               label_map):
    """Initializes an instance of LaserTaggerPredictor.

    Args:
      tf_predictor: Loaded Tensorflow model.
      example_builder: BERT example builder.
      label_map: Mapping from tags to tag IDs.
    """
    self._predictor = tf_predictor
    self._example_builder = example_builder
    self._id_2_tag = {
        tag_id: tagging.Tag(tag) for tag, tag_id in label_map.items()
    }

  def predict(self, sources):
    """Returns realized prediction for given sources."""
    example = self._example_builder.build_bert_example(sources)
    if example is None:
      raise ValueError("Example couldn't be built.")

    # Predict tag IDs.
    keys = ['input_ids', 'input_mask', 'segment_ids']
    out = self._predictor({key: [example.features[key]] for key in keys})
    predicted_ids = out['pred'][0].tolist()
    # Realize output.
    example.features['labels'] = predicted_ids
    # Mask out the begin and the end token.
    example.features['labels_mask'] = [0] + [1] * (len(predicted_ids) - 2) + [0]
    labels = [
        self._id_2_tag[label_id] for label_id in example.get_token_labels()
    ]
    return example.editing_task.realize_output(labels)
