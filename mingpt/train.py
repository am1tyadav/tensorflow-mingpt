from typing import Callable, Iterable

import tensorflow as tf


def data_generator_wrapper(
    data_generator: Callable[[], tuple[tf.Tensor]]
) -> Iterable[tuple[tf.Tensor]]:
    while True:
        yield data_generator()


def train_model(
    model: tf.keras.Model,
    training_generator: Callable[[], tuple[tf.Tensor]],
    validation_generator: Callable[[], tuple[tf.Tensor]],
    epochs: int = 10,
    steps: int = 1000,
    learning_rate: float = 6e-4,
):
    training_gen = data_generator_wrapper(training_generator)
    validation_gen = data_generator_wrapper(validation_generator)

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.SparseCategoricalCrossentropy(),
    )

    _ = model.fit(
        training_gen,
        validation_data=validation_gen,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_steps=int(steps / 10),
        verbose=True,
    )
