from typing import Callable, Iterable

import tensorflow as tf


def train_model(
    model: tf.keras.Model,
    training_generator: Iterable[tuple[tf.Tensor]],
    validation_generator: Iterable[tuple[tf.Tensor]],
    epochs: int = 100,
    steps: int = 500,
    validation_steps: int = 200,
    learning_rate: float = 3e-4,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(),
    )

    _ = model.fit(
        training_generator,
        validation_data=validation_generator,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.EarlyStopping(patience=12),
        ],
        verbose=True,
    )

    return model
