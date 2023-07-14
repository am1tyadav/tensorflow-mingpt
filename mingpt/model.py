import numpy as np
import tensorflow as tf


class AffinityLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        embedding_dim: int,
        block_size: int,
        dropout_rate: float = 0.4,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self._embedding_dim = embedding_dim
        self._dropout_rate = dropout_rate

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
        affinities = tf.nn.dropout(affinities, self._dropout_rate)
        return tf.matmul(affinities, value)


def create_single_head_attention_block(
    head_size: int, block_size: int, embedding_dim: int
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(block_size, embedding_dim))

    key = tf.keras.layers.Dense(head_size, use_bias=False)(
        inputs
    )  # batch_dim, block_size, head_size
    query = tf.keras.layers.Dense(head_size, use_bias=False)(inputs)
    value = tf.keras.layers.Dense(head_size, use_bias=False)(inputs)

    affinities = AffinityLayer(embedding_dim, block_size)([query, key, value])

    return tf.keras.models.Model(inputs, affinities)


def create_multi_head_attention_block(
    num_heads: int,
    head_size: int,
    block_size: int,
    embedding_dim: int,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(block_size, embedding_dim))

    normalised_inputs = tf.keras.layers.LayerNormalization()(inputs)

    outputs = [
        create_single_head_attention_block(head_size, block_size, embedding_dim)(
            normalised_inputs
        )
        for _ in range(0, num_heads)
    ]

    mh_attention_outputs = tf.keras.layers.concatenate(outputs, axis=-1)

    projection = tf.keras.layers.Dense(embedding_dim, activation="relu")(
        mh_attention_outputs
    )
    projection = tf.keras.layers.Dropout(dropout_rate)(projection)
    projection_with_skip_outputs = tf.keras.layers.Add()(
        [projection, inputs]
    )  # inputs are origninal, not normalised
    projection_with_skip_outputs = tf.keras.layers.LayerNormalization()(
        projection_with_skip_outputs
    )

    linear_head_outputs = tf.keras.layers.Dense(4 * embedding_dim, activation="relu")(
        projection_with_skip_outputs
    )
    linear_head_projected = tf.keras.layers.Dense(embedding_dim)(linear_head_outputs)
    linear_head_projected = tf.keras.layers.Dropout(dropout_rate)(linear_head_projected)
    outputs = tf.keras.layers.Add()(
        [linear_head_projected, projection_with_skip_outputs]
    )

    return tf.keras.models.Model(inputs, outputs)


def create_language_model(
    vocab_size: int,
    block_size: int,
    embedding_dim: int,
    num_heads: int,
    num_attention_blocks: int = 3,
    learning_rate=3e-4,
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

    outputs = tf.keras.layers.Add()([token_embeddings, position_embeddings])

    for _ in range(0, num_attention_blocks):
        multi_head_block = create_multi_head_attention_block(
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            embedding_dim=embedding_dim,
        )

        outputs = multi_head_block(outputs)

    outputs = tf.keras.layers.LayerNormalization()(outputs)
    outputs = tf.keras.layers.Dense(vocab_size, name="logits")(outputs)
    outputs = tf.keras.layers.Softmax()(outputs)

    model = tf.keras.models.Model(inputs_tokens, outputs)

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(),
    )

    return model
