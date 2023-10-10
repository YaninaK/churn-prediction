import os
import logging
import tensorflow as tf
import joblib


logger = logging.getLogger()

__all__ = ["store", "load"]


def store(
    lstm_model,
    history,
    filename_model: str,
    filename_history: str,
    path: str = "default",
):
    if path == "default":
        path = models_path()

    filepath_model = os.path.join(path, filename_model + ".keras")

    logger.info(f"Saving model in {filepath_model}")
    lstm_model.save(filepath_model)

    filepath_history = os.path.join(path, filename_history + ".joblib")

    logger.info(f"Dumpung history into {filepath_history}")
    joblib.dump(history, filepath_history)


def load(filename_model: str, path: str = "default"):
    if path == "default":
        path = models_path()

    filepath_model = os.path.join(path, filename_model + ".keras")

    logger.info(f"Loading model from {filepath_model}")
    model = tf.keras.models.load_model(filepath_model)

    return model


def models_path() -> str:
    script_path = os.path.abspath(__file__)
    script_dir_path = os.path.dirname(script_path)
    models_folder = os.path.join(script_dir_path, "..", "..", "..", "models")

    return models_folder
