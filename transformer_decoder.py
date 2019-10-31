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
"""Transformer decoder."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Any, Mapping, Text

import tensorflow as tf

from official_transformer import attention_layer
from official_transformer import embedding_layer
from official_transformer import ffn_layer
from official_transformer import model_utils
from official_transformer import transformer


class TransformerDecoder(transformer.Transformer):
  """Transformer decoder.

  Attributes:
    train: Whether the model is in training mode.
    params: Model hyperparameters.
  """

  def __init__(self, params, train):
    """Initializes layers to build Transformer model.

    Args:
      params: hyperparameter object defining layer sizes, dropout values, etc.
      train: boolean indicating whether the model is in training mode. Used to
        determine if dropout layers should be added.
    """
    self.train = train
    self.params = params
    self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
        params["vocab_size"], params["hidden_size"],
        method="matmul" if params["use_tpu"] else "gather")

    if self.params["use_full_attention"]:
      self.decoder_stack = transformer.DecoderStack(params, train)
    else:
      self.decoder_stack = DecoderStack(params, train)

  def __call__(self, inputs, encoder_outputs, targets=None):
    """Calculates target logits or inferred target sequences.

    Args:
      inputs: int tensor with shape [batch_size, input_length].
      encoder_outputs: int tensor with shape
        [batch_size, input_length, hidden_size]
      targets: None or int tensor with shape [batch_size, target_length].

    Returns:
      If targets is defined, then return logits for each word in the target
      sequence. float tensor with shape [batch_size, target_length, vocab_size]
      If target is none, then generate output sequence one token at a time.
        returns a dictionary {
          output: [batch_size, decoded length]
          score: [batch_size, float]}
    """
    # Variance scaling is used here because it seems to work in many problems.
    # Other reasonable initializers may also work just as well.
    initializer = tf.variance_scaling_initializer(
        self.params["initializer_gain"], mode="fan_avg", distribution="uniform")
    with tf.variable_scope("Transformer", initializer=initializer):
      # Calculate attention bias for encoder self-attention and decoder
      # multi-headed attention layers.
      attention_bias = model_utils.get_padding_bias(inputs)

      # Generate output sequence if targets is None, or return logits if target
      # sequence is known.
      if targets is None:
        return self.predict(encoder_outputs, attention_bias)
      else:
        logits = self.decode(targets, encoder_outputs, attention_bias)
        return logits

  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""

    timing_signal = model_utils.get_position_encoding(
        max_decode_length + 1, self.params["hidden_size"])
    decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
        max_decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids: Current decoded sequences.
          int tensor with shape [batch_size * beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      # Set decoder input to the last generated IDs
      decoder_input = ids[:, -1:]

      # Preprocess decoder input by getting embeddings and adding timing signal.
      decoder_input = self.embedding_softmax_layer(decoder_input)
      decoder_input += timing_signal[i:i + 1]

      self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
      if self.params["use_full_attention"]:
        encoder_outputs = cache.get("encoder_outputs")
      else:
        encoder_outputs = cache.get("encoder_outputs")[:, i:i+1]
      decoder_outputs = self.decoder_stack(
          decoder_input, encoder_outputs, self_attention_bias,
          cache.get("encoder_decoder_attention_bias"), cache)
      logits = self.embedding_softmax_layer.linear(decoder_outputs)
      logits = tf.squeeze(logits, axis=[1])
      return logits, cache
    return symbols_to_logits_fn


class DecoderStack(tf.layers.Layer):
  """Modified Transformer decoder stack.

  Like the standard Transformer decoder stack but:
    1. Removes the encoder-decoder attention layer, and
    2. Adds a layer to project the concatenated [encoder activations, hidden
       state] to the hidden size.
  """

  def __init__(self, params, train):
    super(DecoderStack, self).__init__()
    self.layers = []
    for _ in range(params["num_hidden_layers"]):
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"], train)
      feed_forward_network = ffn_layer.FeedFowardNetwork(  # NOTYPO
          params["hidden_size"], params["filter_size"],
          params["relu_dropout"], train, params["allow_ffn_pad"])

      proj_layer = tf.layers.Dense(
          params["hidden_size"], use_bias=True, name="proj_layer")

      self.layers.append([
          transformer.PrePostProcessingWrapper(
              self_attention_layer, params, train),
          transformer.PrePostProcessingWrapper(
              feed_forward_network, params, train),
          proj_layer])

    self.output_normalization = transformer.LayerNormalization(
        params["hidden_size"])

  def call(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
           attention_bias=None, cache=None):
    """Returns the output of the decoder layer stacks.

    Args:
      decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
      encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
      decoder_self_attention_bias: bias for decoder self-attention layer.
        [1, 1, target_len, target_length]
      attention_bias: bias for encoder-decoder attention layer.
        [batch_size, 1, 1, input_length]
      cache: (Used for fast decoding) A nested dictionary storing previous
        decoder self-attention values. The items are:
          {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                     "v": tensor with shape [batch_size, i, value_channels]},
           ...}

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """
    for n, layer in enumerate(self.layers):
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]
      proj_layer = layer[2]

      decoder_inputs = tf.concat([decoder_inputs, encoder_outputs], axis=-1)
      decoder_inputs = proj_layer(decoder_inputs)

      # Run inputs through the sublayers.
      layer_name = "layer_%d" % n
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          decoder_inputs = self_attention_layer(
              decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
        with tf.variable_scope("ffn"):
          decoder_inputs = feed_forward_network(decoder_inputs)

    return self.output_normalization(decoder_inputs)
