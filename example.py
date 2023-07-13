import os

from loguru import logger

import mingpt


def main():
    filepath = "tmp/data.txt"
    batch_size = 16
    block_size = 8
    embedding_dim = 64
    num_heads = 4

    if not os.path.isfile(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mingpt.data.download_example_data(filepath=filepath)

    raw_data = mingpt.data.load_data(filepath=filepath)
    vocab, encoder, decoder = mingpt.data.create_vocab(data=raw_data)

    train_data, valid_data = mingpt.data.create_dataset(encoder(raw_data))
    train_generator = mingpt.data.batch_generator(
        data=train_data, batch_size=batch_size, block_size=block_size
    )
    valid_generator = mingpt.data.batch_generator(
        data=valid_data, batch_size=batch_size, block_size=block_size
    )

    model = mingpt.model.create_language_model(
        len(vocab), block_size, embedding_dim, num_heads
    )
    sequence = mingpt.generate.generate_sequence(
        model, "Firs", encoder, decoder, block_size
    )

    logger.info("Before training:")
    logger.info(sequence)

    mingpt.train.train_model(model, train_generator, valid_generator)

    sequence = mingpt.generate.generate_sequence(
        model, "Firs", encoder, decoder, block_size
    )

    logger.info("After training")
    logger.info(sequence)


if __name__ == "__main__":
    main()
