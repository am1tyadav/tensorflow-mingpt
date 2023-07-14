import numpy as np
import tensorflow as tf


def generate_sequence(
    model: tf.keras.Model,
    tokens: list[int],
    block_size: int,
    num_tokens_to_generate: int = 100,
) -> list[int]:
    tokens = tf.convert_to_tensor(tokens, dtype=tf.int64)

    generation_model = tf.keras.models.Model(
        model.input, model.get_layer("logits").output
    )

    for _ in range(0, num_tokens_to_generate):
        inputs = tokens[-block_size:]
        inputs = tf.expand_dims(inputs, axis=0)

        predictions = generation_model.predict(inputs, verbose=False)
        prediction = predictions[:, -1, :]

        next_index = tf.raw_ops.Multinomial(logits=prediction, num_samples=1)

        tokens = tf.concat([tokens, next_index[0]], axis=-1)

    return np.array(tokens).tolist()
