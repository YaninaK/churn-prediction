import logging
import tensorflow as tf


logger = logging.getLogger(__name__)

__all__ = ["generate_LSTM_model"]


METRICS = [
    tf.keras.metrics.BinaryCrossentropy(name="cross entropy"),
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


def get_LSTM_model(
    input_sequence_length: int,
    n_features: int,
    n_units: int,
    output_bias=None,
    metrics=None,
):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    if metrics is None:
        metrics = METRICS

    lstm_inputs = tf.keras.layers.Input(
        shape=(input_sequence_length, n_features), name="lstm_inputs"
    )
    lstm_output = tf.keras.layers.LSTM(n_units, name="lstm_output")(lstm_inputs)

    outputs = tf.keras.layers.Dense(
        1,
        activation=tf.keras.activations.sigmoid,
        name="outputs",
        bias_initializer=output_bias,
    )(lstm_output)

    model = tf.keras.models.Model(lstm_inputs, outputs, name="lstm_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model


def get_ts_embeddings_model(model):
    sharing_input = model.get_layer("lstm_inputs").get_output_at(0)
    sharing_output = model.get_layer("lstm_output").get_output_at(0)
    emb_model = tf.keras.Model(sharing_input, sharing_output)

    return emb_model


def get_ts_embeddings(df, seq, emb_model):
    emb_lstm = emb_model.predict(seq)
    cols = [f"ts_emb_{i}" for i in range(emb_lstm.shape[1])]

    return pd.DataFrame(emb_lstm, index=df.index, columns=cols)
