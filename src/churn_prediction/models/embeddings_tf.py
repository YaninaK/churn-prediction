import logging
import pandas as pd
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


def transform_embeddings(df, feature, output_dim, lookup_and_embed):
    emb = lookup_and_embed(tf.constant(df[feature])).numpy()
    cols = [f"{feature}_{i}" for i in range(output_dim)]

    return pd.DataFrame(emb, index=df.index, columns=cols)
