import numpy as np
import tensorflow as tf


class AffinityLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim: int,
        block_size: int,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self._embedding_dim = embedding_dim
        self._tril = tf.linalg.LinearOperatorLowerTriangular(
            tf.cast(tf.ones((block_size, block_size)), dtype=tf.bool)
        ).to_dense()

    def call(self, inputs: tuple[tf.Tensor]) -> tf.Tensor:
        query, key, value = inputs
        # both are of shape batch_dim, block_size, head_size

        key = tf.transpose(key, perm=[0, 2, 1])
        # batch dot product
        affinities = tf.matmul(query, key)
        # The following weighting is done to retain the variance
        affinities = affinities * self._embedding_dim**-0.5
        # At this point affinities is of shape batch_size, block_size, block size
        affinities = tf.where(
            self._tril, affinities, tf.ones(self._tril.shape) * np.inf * -1
        )
        affinities = tf.nn.softmax(affinities)
        # Softmax is applied on the last dim, shape remains the same: batch_size, block_size, block size
        # dot product between this and value will give us the final affinities, return batch, block, head
        return tf.matmul(affinities, value)


def create_single_head_attention_block(
    head_size: int, block_size: int, embedding_dim: int
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(block_size, embedding_dim))

    key = tf.keras.layers.Dense(head_size)(inputs)  # batch_dim, block_size, head_size
    query = tf.keras.layers.Dense(head_size)(inputs)
    value = tf.keras.layers.Dense(head_size)(inputs)

    affinities = AffinityLayer(embedding_dim, block_size)([query, key, value])
    return tf.keras.models.Model(inputs, affinities)


def create_multi_head_attention_block(
    num_heads: int, head_size: int, block_size: int, embedding_dim: int
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(block_size, embedding_dim))

    outputs = [
        create_single_head_attention_block(head_size, block_size, embedding_dim)(inputs)
        for _ in range(0, num_heads)
    ]

    outputs = tf.keras.layers.concatenate(outputs, axis=-1)
    return tf.keras.models.Model(inputs, outputs)


def create_language_model(
    vocab_size: int, block_size: int, embedding_dim: int, num_heads: int
) -> tf.keras.Model:
    head_size = embedding_dim // num_heads

    inputs_tokens = tf.keras.layers.Input(shape=(block_size,))

    token_embeddings = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim
    )(inputs_tokens)
    position_embeddings = tf.keras.layers.Embedding(
        input_dim=block_size, output_dim=embedding_dim
    )(tf.range(0, block_size))
    position_embeddings = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=0))(
        position_embeddings
    )

    embeddings = tf.keras.layers.Add()([token_embeddings, position_embeddings])

    multi_head_block = create_multi_head_attention_block(
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        embedding_dim=embedding_dim,
    )

    outputs = multi_head_block(embeddings)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(outputs)
    return tf.keras.models.Model(inputs_tokens, outputs)
