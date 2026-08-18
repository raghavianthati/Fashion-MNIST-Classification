import tensorflow as tf
import numpy as np

print("Loading Fashion MNIST...")

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Use only 10,000 training images for quick training
x_train = x_train[:10000]
y_train = y_train[:10000]

x_test = x_test[:2000]
y_test = y_test[:2000]

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("Training images:", len(x_train))
print("Testing images:", len(x_test))

# Create simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nTraining started...\n")

# Train
model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64
)

# Test
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n==============================")
print("FASHION MNIST RESULT")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

# Prediction
predictions = model.predict(x_test[:1], verbose=0)

predicted = np.argmax(predictions[0])

print("Actual:", class_names[y_test[0]])
print("Predicted:", class_names[predicted])

print("==============================")

# Save model
model.save("fashion_model.keras")

print("\nModel saved successfully!")