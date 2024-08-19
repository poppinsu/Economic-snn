import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data from Yahoo Finance
data = yf.download('NVDA', start='2020-01-01', end='2023-01-01')
data = data['Close'].values

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))


# Create dataset for supervised learning
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


look_back = 10
X, Y = create_dataset(data, look_back)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize network parameters
input_size = look_back
hidden_size1 = 20  # Increased hidden size
hidden_size2 = 10  # Additional hidden layer
output_size = 1
learning_rate = 0.01
iterations = 2000  # Increased iterations for better convergence

# Initialize weights
W1 = np.random.randn(input_size, hidden_size1)
W2 = np.random.randn(hidden_size1, hidden_size2)
W3 = np.random.randn(hidden_size2, output_size)


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Training process
losses = []
for i in range(iterations):
    # Forward propagation
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3)
    A3 = sigmoid(Z3)

    # Error calculation (Mean Squared Error)
    error = y_train.reshape(-1, 1) - A3
    loss = np.mean(np.square(error))
    losses.append(loss)

    # Backpropagation
    dA3 = error * sigmoid_derivative(A3)
    dA2 = dA3.dot(W3.T) * sigmoid_derivative(A2)
    dA1 = dA2.dot(W2.T) * sigmoid_derivative(A1)

    W3 += A2.T.dot(dA3) * learning_rate
    W2 += A1.T.dot(dA2) * learning_rate
    W1 += X_train.T.dot(dA1) * learning_rate


# Evaluation
def predict(X):
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    Z3 = np.dot(A2, W3)
    return sigmoid(Z3)


predictions = predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
rmse = np.sqrt(np.mean((predictions - y_test_scaled) ** 2))
print(f"RMSE: {rmse}")

# Print results
print("Predicted values:", predictions[:5])
print("Actual values:", y_test_scaled[:5])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(predictions, label='Predicted')
plt.plot(y_test_scaled, label='Actual')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Predicted vs Actual Stock Prices')
plt.legend()
plt.show()

# Plot loss over iterations
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.show()
