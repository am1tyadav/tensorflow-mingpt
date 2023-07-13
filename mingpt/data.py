"""Load and preprocess data for minGPT."""

from pathlib import Path
from typing import Callable

import requests
import tensorflow as tf
from loguru import logger


def download_example_data(filepath: Path):
    """
    Downloads example data from a given URL and saves it to the specified filepath.

    Args:
    ----
        filepath (Path): The path to save the downloaded file.

    Returns:
    -------
        None

    Example:
    -------
        >>> download_example_data(Path("data/input.txt"))
    """
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    timeout = 1200

    response = requests.get(url=url, timeout=timeout)
    content = response.content.decode()

    with open(filepath, "w") as _file:
        _file.write(content)

    logger.info(f"File downloaded to {filepath}")


def load_data(filepath: Path) -> str:
    """
    Loads data from a given file.

    Args:
    ----
        filepath (Path): The path of the file to load.

    Returns:
    -------
        str: The loaded data as a string.

    Example:
    -------
        >>> data = load_data(Path("data/input.txt"))
    """
    data = None

    with open(filepath, "r") as _file:
        data = _file.read()

    logger.info(f"Data loaded from {filepath}")
    return data


def _encoder(char_to_index: dict[str, int]) -> Callable[[str], list[int]]:
    """
    Returns an encoding function that converts text into a list of encoded tokens.

    Args:
    ----
        char_to_index (dict[str, int]): A dictionary mapping characters to their corresponding indices.

    Returns:
    -------
        Callable[[str], list[str]]: The encoding function.

    Example:
    -------
        >>> encoder = _encoder(char_to_index)
        >>> encoded_text = encoder("Hello")
    """

    def _encode_text(text: str) -> list[int]:
        return [char_to_index[char] for char in text]

    return _encode_text


def _decoder(index_to_char: dict[int, str]) -> Callable[[list[int]], str]:
    """
    Returns a decoding function that converts a list of tokens into text.

    Args:
    ----
        index_to_char (dict[int, str]): A dictionary mapping indices to their corresponding characters.

    Returns:
    -------
        Callable[[list[int]], str]: The decoding function.

    Example:
    -------
        >>> decoder = _decoder(index_to_char)
        >>> decoded_text = decoder([0, 1, 2, 3, 4])
    """

    def _decode_tokens(tokens: list[int]) -> str:
        return "".join([index_to_char[token] for token in tokens])

    return _decode_tokens


def create_vocab(
    data: str,
) -> tuple[list[str], Callable[[str], list[str]], Callable[[list[int]], str]]:
    """
    Creates a vocabulary and returns the vocabulary list, an encoding function, and a decoding function.

    Args:
    ----
        data (str): The text data to create the vocabulary from.

    Returns:
    -------
        tuple[list[str], Callable[[str], list[str]], Callable[[list[int]], str]]: A tuple containing the vocabulary list,
        the encoding function, and the decoding function.

    Example:
    -------
        >>> vocab, encoder, decoder = create_vocab("Hello World!")
    """
    vocab = sorted(list(set(data)))
    char_to_index = {char: index for index, char in enumerate(vocab)}
    index_to_char = {value: key for key, value in char_to_index.items()}
    encoder = _encoder(char_to_index)
    decoder = _decoder(index_to_char)

    logger.info("Vocab, encoder and decoder created")

    return vocab, encoder, decoder


def create_dataset(tokens: list[int], split: float = 0.9) -> tuple[tf.Tensor]:
    if split < 0.0 or split > 1.0:
        raise AssertionError("split must be between 0 and 1")

    num_training_examples = int(len(tokens) * split)

    training_data = tokens[:num_training_examples]
    validation_data = tokens[num_training_examples:]

    training_data = tf.convert_to_tensor(training_data, dtype=tf.int32)
    validation_data = tf.convert_to_tensor(validation_data, dtype=tf.int32)

    return training_data, validation_data


def batch_generator(
    data: tf.Tensor, batch_size: int, block_size: int
) -> Callable[[], tuple[tf.Tensor]]:
    num_examples = len(data)

    def _batch_generator() -> tuple[tf.Tensor]:
        indices = tf.random.uniform(
            shape=(batch_size,), maxval=num_examples - block_size, dtype=tf.int32
        )
        examples = tf.stack([data[index : index + block_size] for index in indices])
        labels = tf.stack(
            [data[index + 1 : index + block_size + 1] for index in indices]
        )
        return examples, labels

    return _batch_generator
