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

import run_lasertagger

import tensorflow as tf


class RunLasertaggerTest(tf.test.TestCase):

  def test_step_calculation(self):
    num_examples = 10
    batch_size = 2
    num_epochs = 3
    warmup_proportion = 0.5
    steps, warmup_steps = run_lasertagger._calculate_steps(
        num_examples, batch_size, num_epochs, warmup_proportion)
    self.assertEqual(steps, 15)
    self.assertEqual(warmup_steps, 7)


if __name__ == '__main__':
  tf.test.main()
