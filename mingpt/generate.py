from typing import Callable

import numpy as np
import tensorflow as tf


def generate_sequence(
    model: tf.keras.Model,
    inputs: str,
    encoder: Callable[[str], list[int]],
    decoder: Callable[[list[int]], str],
    block_size: int,
    num_tokens_to_generate: int = 100,
) -> str:
    tokens = encoder(inputs)

    if len(tokens) < block_size:
        pad = [0] * (block_size - len(tokens))
        tokens = pad + tokens

    for _ in range(0, num_tokens_to_generate):
        inputs_sliced = tf.convert_to_tensor(tokens[-block_size:], dtype=tf.int32)
        predictions = model.predict(
            tf.expand_dims(inputs_sliced, axis=0), verbose=False
        )
        prediction = predictions[:, -1, :]

        next_index = tf.raw_ops.Multinomial(
            logits=prediction, num_samples=1, output_dtype=tf.int32
        )

        tokens.append(int(np.squeeze(next_index)))
        inputs_sliced = tf.concat([inputs_sliced, next_index[0]], axis=-1)

    return decoder(tokens)
