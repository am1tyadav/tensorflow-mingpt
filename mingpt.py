import os
from typing import Annotated

import tensorflow as tf
import typer
from loguru import logger

import mingpt

app = typer.Typer(name="mingpt")


FilePathOption = Annotated[
    str,
    typer.Option(
        help="Path to data file. It will be downloaded if it doesn't exist already."
    ),
]

ModelPathOption = Annotated[
    str, typer.Option(help="Path to the model checkpoint in tf.keras format.")
]

BatchSizeOption = Annotated[int, typer.Option(help="Batch size for training.")]
BlockSizeOption = Annotated[
    int, typer.Option(help="Block size i.e. number of tokens to be used as context.")
]
EmbeddingDimOption = Annotated[
    int, typer.Option(help="Embedding dimension used for the attention blocks.")
]
NumHeadsOption = Annotated[
    int, typer.Option(help="Number of attention heads in a single attention block.")
]
NumAttentionBlocksOption = Annotated[
    int, typer.Option(help="Number of attention blocks to be used.")
]


def ensure_load_data(filepath: str) -> str:
    if not os.path.isfile(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mingpt.data.download_example_data(filepath=filepath)
    return mingpt.data.load_data(filepath=filepath)


@app.command()
def generate(
    text: Annotated[str, typer.Argument(help="Starting text")],
    use_pretrained: Annotated[
        bool, typer.Argument(help="If to use the pretrained model or not")
    ] = True,
    filepath: FilePathOption = "./tmp/data.txt",
    model_filepath: ModelPathOption = "./tmp/mingpt.h5",
    block_size: BlockSizeOption = 256,
    embedding_dim: EmbeddingDimOption = 384,
    num_heads: NumHeadsOption = 6,
    num_attention_blocks: NumAttentionBlocksOption = 6,
):
    raw_data = ensure_load_data(filepath)
    vocab, encoder, decoder = mingpt.data.create_vocab(raw_data)

    model = mingpt.model.create_language_model(
        len(vocab), block_size, embedding_dim, num_heads, num_attention_blocks
    )

    if use_pretrained:
        model.load_weights(model_filepath)
        logger.info("Model checkpoint loaded")

    input_tokens = encoder(text)
    num_tokens = len(input_tokens)

    if num_tokens > block_size:
        input_tokens = input_tokens[-block_size:]
    elif num_tokens < block_size:
        input_tokens = [encoder(" ")[0]] * (block_size - num_tokens) + input_tokens

    sequence = mingpt.generate.generate_sequence(
        model=model, tokens=input_tokens, block_size=block_size
    )

    logger.info(decoder(sequence))


@app.command()
def train(
    filepath: FilePathOption = "tmp/data.txt",
    model_filepath: ModelPathOption = "./tmp/mingpt.h5",
    batch_size: BatchSizeOption = 16,
    block_size: BlockSizeOption = 256,
    embedding_dim: EmbeddingDimOption = 384,
    num_heads: NumHeadsOption = 6,
    num_attention_blocks: NumAttentionBlocksOption = 6,
    allow_pretrained: Annotated[
        bool, typer.Option(help="Allow loading pretrained models")
    ] = True,
):
    raw_data = ensure_load_data(filepath)
    vocab, encoder, decoder = mingpt.data.create_vocab(raw_data)

    train_data, valid_data = mingpt.data.create_dataset(encoder(raw_data))
    train_generator = mingpt.data.batch_generator(
        data=train_data, batch_size=batch_size, block_size=block_size
    )
    valid_generator = mingpt.data.batch_generator(
        data=valid_data, batch_size=batch_size, block_size=block_size
    )

    num_training_examples = len(train_generator) // batch_size
    num_validation_examples = len(valid_generator) // batch_size

    mirrored_strategy = tf.distribute.MirroredStrategy()

    num_gpus = mirrored_strategy.num_replicas_in_sync

    logger.info(f"Number of devices: {num_gpus}")

    train_generator = train_generator.batch(num_gpus * batch_size).repeat()
    valid_generator = valid_generator.batch(num_gpus * batch_size).repeat()

    train_generator = mirrored_strategy.experimental_distribute_dataset(train_generator)
    valid_generator = mirrored_strategy.experimental_distribute_dataset(valid_generator)

    with mirrored_strategy.scope():
        model = mingpt.model.create_language_model(
            len(vocab), block_size, embedding_dim, num_heads, num_attention_blocks
        )

        if allow_pretrained and os.path.isfile(model_filepath):
            model.load_weights(model_filepath)
            logger.info("Model checkpoint loaded")

    logger.info(model.summary())

    _ = mingpt.train.train_model(
        model,
        model_filepath,
        train_generator,
        valid_generator,
        steps=num_training_examples,
        validation_steps=num_validation_examples,
    )


if __name__ == "__main__":
    app()