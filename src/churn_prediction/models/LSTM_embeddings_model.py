import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ["generate_LSTM_model"]


METRICS = [
    tf.keras.metrics.BinaryCrossentropy(name="cross entropy"),  # same as model's loss
    tf.keras.metrics.MeanSquaredError(name="Brier score"),
    tf.keras.metrics.TruePositives(name="tp"),
    tf.keras.metrics.FalsePositives(name="fp"),
    tf.keras.metrics.TrueNegatives(name="tn"),
    tf.keras.metrics.FalseNegatives(name="fn"),
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.AUC(name="prc", curve="PR"),  # precision-recall curve
]


def get_model(
    input_sequence_length: int,
    n_features: int,
    n_units: int,
    vocab_s: list,
    n_labels_s: int,
    embedding_size_s: int,
    vocab_e: list,
    vocab_o: list,
    n_features_other: int,
    n_units_others: int,
    n_units_all: int,
    output_bias=None,
    metrics=None,
):
    """
    LSTM parameters: input_sequence_length, n_features, n_units

    'state' embeddings parameters:
    vocab_s - a list if unique labels of feature 'state'
    n_labels_s = the number of unique labels of feature 'state'
    embedding_size_s - embedding size of feature 'state'

    'education' and 'occupation' parameters:
    vocab_e - a list if unique labels of feature 'education'
    vocab_o - a list if unique labels of feature 'occupation'

    other selected features parameters:
    n_features_other - the number of selected features
    n_units_others - the number of units in Dense layer of other selected features

    n_units_all - the number of units in Dense layer before the final layer
    output_bias - initial bias if classes are imbalanced
    metrics - metrics of the model to be collected in training and validation.

    """
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    if metrics is None:
        metrics = METRICS

    # LSTM
    lstm_inputs = tf.keras.layers.Input(
        shape=(input_sequence_length, n_features), name="lstm_inputs"
    )
    lstm_output = tf.keras.layers.LSTM(n_units, name="lstm_output")(lstm_inputs)

    # states
    inputs_s = tf.keras.layers.Input(shape=(), name="state_inputs", dtype=tf.string)
    str_lookup_s = tf.keras.layers.StringLookup(
        vocabulary=vocab_s, name="string_lookup_state"
    )(inputs_s)
    embeddings_s = tf.keras.layers.Embedding(
        n_labels_s + 1, embedding_size_s, input_length=1, name="embeddings_state"
    )(str_lookup_s)
    flatten_s = tf.keras.layers.Flatten(name="flatten_state")(embeddings_s)

    # education, occupation
    inputs_e = tf.keras.layers.Input(shape=(), name="education_inputs", dtype=tf.string)
    str_lookup_layer_e = tf.keras.layers.StringLookup(
        vocabulary=vocab_e, output_mode="one_hot", name="str_lookup_layer_education"
    )(inputs_e)

    inputs_o = tf.keras.layers.Input(
        shape=(), name="occupation_inputs", dtype=tf.string
    )
    str_lookup_layer_o = tf.keras.layers.StringLookup(
        vocabulary=vocab_o, output_mode="one_hot", name="str_lookup_layer_occupation"
    )(inputs_o)

    # other_features
    inputs_all = tf.keras.layers.Input(shape=(n_features_other), name="inputs_all")
    dense_others = tf.keras.layers.Dense(
        n_units_all,
        activation=tf.keras.activations.gelu,
        name="dense_others",
    )(inputs_all)

    # all
    concat_all = tf.keras.layers.Concatenate(axis=-1, name="concat_all")(
        [lstm_output, flatten_s, str_lookup_layer_e, str_lookup_layer_o, dense_others]
    )
    dense_all = tf.keras.layers.Dense(
        n_units_all,
        activation=tf.keras.activations.gelu,
        name="dense_all",
    )(concat_all)

    # output
    outputs = tf.keras.layers.Dense(
        1,
        activation=tf.keras.activations.sigmoid,
        name="outputs",
        bias_initializer=output_bias,
    )(concat_all)

    model = tf.keras.models.Model(
        [lstm_inputs, inputs_s, inputs_e, inputs_o, inputs_all],
        outputs,
        name="lstm_model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model
