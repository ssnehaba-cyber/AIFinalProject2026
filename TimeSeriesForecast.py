
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# =============================
# 1. DATA GENERATION
# =============================
def generate_time_series(n_points=3000):
    np.random.seed(42)
    t = np.arange(n_points)

    trend = 0.0005 * t
    seasonality1 = 0.5 * np.sin(2 * np.pi * t / 24)     # daily
    seasonality2 = 0.3 * np.sin(2 * np.pi * t / 168)    # weekly
    seasonality3 = 0.2 * np.sin(2 * np.pi * t / 365)    # yearly
    noise = 0.1 * np.random.randn(n_points)

    series = trend + seasonality1 + seasonality2 + seasonality3 + noise
    return series.reshape(-1, 1)


# =============================
# 2. PREPROCESSING
# =============================
def create_sequences(data, window_size=30, horizon=5):
    X, y = [], []
    for i in range(len(data) - window_size - horizon):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + horizon])
    return np.array(X), np.array(y)


# =============================
# 3. BASELINE LSTM MODEL
# =============================
def build_lstm_model(input_shape, horizon):
    model = models.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(horizon)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# =============================
# 4. ATTENTION MODEL
# =============================
class SelfAttention(layers.Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context_vector, attention_weights


def build_attention_lstm(input_shape, horizon):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    context_vector, attention_weights = SelfAttention()(x)
    outputs = layers.Dense(horizon)(context_vector)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# =============================
# 5. MAIN PIPELINE
# =============================
def main():
    # Generate data
    series = generate_time_series()

    # Scale
    scaler = MinMaxScaler()
    scaled_series = scaler.fit_transform(series)

    # Create sequences
    WINDOW_SIZE = 30
    HORIZON = 5

    X, y = create_sequences(scaled_series, WINDOW_SIZE, HORIZON)

    # Train/Val/Test Split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # =====================
    # BASELINE MODEL
    # =====================
    print("\nTraining Baseline LSTM Model...")
    baseline_model = build_lstm_model((WINDOW_SIZE, 1), HORIZON)
    baseline_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    baseline_preds = baseline_model.predict(X_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_test.flatten(), baseline_preds.flatten()))
    baseline_mae = mean_absolute_error(y_test.flatten(), baseline_preds.flatten())

    # =====================
    # ATTENTION MODEL
    # =====================
    print("\nTraining Attention LSTM Model...")
    attention_model = build_attention_lstm((WINDOW_SIZE, 1), HORIZON)
    attention_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

    attention_preds = attention_model.predict(X_test)
    attention_rmse = np.sqrt(mean_squared_error(y_test.flatten(), attention_preds.flatten()))
    attention_mae = mean_absolute_error(y_test.flatten(), attention_preds.flatten())

    # =====================
    # RESULTS
    # =====================
    print("\n========== PERFORMANCE COMPARISON ==========")
    print(f"Baseline LSTM RMSE: {baseline_rmse:.4f}")
    print(f"Baseline LSTM MAE : {baseline_mae:.4f}")
    print("-------------------------------------------")
    print(f"Attention LSTM RMSE: {attention_rmse:.4f}")
    print(f"Attention LSTM MAE : {attention_mae:.4f}")
    print("===========================================\n")

    # =====================
    # ATTENTION WEIGHT INSPECTION
    # =====================
    attention_layer_model = models.Model(
        inputs=attention_model.input,
        outputs=attention_model.layers[2].output
    )

    context_vector, attention_weights = attention_layer_model.predict(X_test[:1])

    print("Sample Attention Weights (for first test sample):")
    print(attention_weights.reshape(-1))

    plt.figure(figsize=(10, 4))
    plt.plot(attention_weights.reshape(-1))
    plt.title("Attention Weights Over Time Window")
    plt.xlabel("Time Step")
    plt.ylabel("Weight")
    plt.show()


if __name__ == "__main__":
    main()
