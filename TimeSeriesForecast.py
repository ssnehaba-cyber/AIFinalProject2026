import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Layer


# 1. Generate Time Series Data
np.random.seed(42)
t = np.arange(0, 1000)
data = (np.sin(0.02 * t) + 0.5 * np.sin(0.05 * t) + 0.2 * np.random.randn(len(t)))

df = pd.DataFrame({"value": data})

# 2. Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 3. Create Sequences
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)

# Train-test split
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Baseline LSTM Model
baseline_model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
baseline_model.compile(optimizer='adam', loss='mse')
baseline_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

baseline_preds = baseline_model.predict(X_test)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_mae = mean_absolute_error(y_test, baseline_preds)

# 5. Attention Layer
class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def call(self, inputs):
        score = tf.matmul(inputs, inputs, transpose_b=True)
        weights = tf.nn.softmax(score, axis=-1)
        context = tf.matmul(weights, inputs)
        return tf.reduce_sum(context, axis=1)

# 6. Attention-based LSTM Model
input_layer = Input(shape=(X_train.shape[1], 1))
lstm_out = LSTM(64, return_sequences=True)(input_layer)
attention_out = Attention()(lstm_out)
output = Dense(1)(attention_out)

attention_model = Model(inputs=input_layer, outputs=output)
attention_model.compile(optimizer='adam', loss='mse')
attention_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

attention_preds = attention_model.predict(X_test)
attention_rmse = np.sqrt(mean_squared_error(y_test, attention_preds))
attention_mae = mean_absolute_error(y_test, attention_preds)

print("Baseline RMSE:", baseline_rmse)
print("Baseline MAE:", baseline_mae)
print("Attention RMSE:", attention_rmse)
print("Attention MAE:", attention_mae)
