from __future__ import annotations

from typing import Any


def build_lstm_model(window_size: int, n_features: int, config: dict[str, Any]):
    import tensorflow as tf

    model_config = config.get("model", config)
    learning_rate = float(model_config.get("learning_rate", 0.001))
    loss = str(model_config.get("loss", "mse"))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, n_features)),
            tf.keras.layers.LSTM(int(model_config.get("lstm_units_1", 64)), return_sequences=True),
            tf.keras.layers.Dropout(float(model_config.get("dropout", 0.2))),
            tf.keras.layers.LSTM(int(model_config.get("lstm_units_2", 32))),
            tf.keras.layers.Dropout(float(model_config.get("dropout", 0.2))),
            tf.keras.layers.Dense(int(model_config.get("dense_units", 16)), activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber() if loss.lower() == "huber" else loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
