import tensorflow as tf


def train_model(
    model: tf.keras.Model,
    model_filepath: str,
    training_generator: tf.data.Dataset,
    validation_generator: tf.data.Dataset,
    epochs: int = 1000,
    steps: int = 500,
    validation_steps: int = 200,
):
    _ = model.fit(
        training_generator,
        validation_data=validation_generator,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=6e-5),
            tf.keras.callbacks.EarlyStopping(patience=7),
            tf.keras.callbacks.ModelCheckpoint(model_filepath, save_best_only=True),
        ],
        verbose=True,
    )
