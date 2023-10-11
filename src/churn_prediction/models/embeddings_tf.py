import logging
import pandas as pd
import numpy as np
import tensorflow as tf


logger = logging.getLogger(__name__)

__all__ = ["generate_tf_embeddings"]


def fit_transform_embeddings(
    df: pd.DataFrame,
    feature: str,
    output_dim: int,
):
    vocab = df[feature].unique().tolist()
    str_lookup_layer = tf.keras.layers.StringLookup(vocabulary=vocab)
    lookup_and_embed = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=[], dtype=tf.string),
            str_lookup_layer,
            tf.keras.layers.Embedding(
                input_dim=str_lookup_layer.vocabulary_size(),
                output_dim=output_dim,
            ),
        ]
    )
    emb = lookup_and_embed(tf.constant(df[feature])).numpy()
    cols = [f"{feature}_{i}" for i in range(output_dim)]

    return pd.DataFrame(emb, index=df.index, columns=cols), lookup_and_embed


def transform_embeddings(
    df: pd.DataFrame, feature: str, output_dim: int, lookup_and_embed
) -> pd.DataFrame:
    emb = lookup_and_embed(tf.constant(df[feature])).numpy()
    cols = [f"{feature}_{i}" for i in range(output_dim)]

    return pd.DataFrame(emb, index=df.index, columns=cols)


def fit_transform_one_hot_encoding(df: pd.DataFrame, feature: str):
    vocab = df[feature].unique().tolist()
    str_lookup_layer = tf.keras.layers.StringLookup(
        vocabulary=vocab, output_mode="one_hot", name="str_lookup_layer"
    )
    arr = str_lookup_layer(df[feature]).numpy()[:, 1:]

    return pd.DataFrame(arr, index=df.index, columns=vocab), str_lookup_layer


def transform_one_hot_encoding(
    df: pd.DataFrame, feature: str, str_lookup_layer
) -> pd.DataFrame:
    arr = str_lookup_layer(df[feature]).numpy()[:, 1:]

    return pd.DataFrame(
        arr, index=df.index, columns=str_lookup_layer.get_vocabulary()[1:]
    )
