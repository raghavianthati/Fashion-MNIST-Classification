import tensorflow as tf
import numpy as np

print("Loading trained model...")

# Load trained model
model = tf.keras.models.load_model("fashion_model.keras")

print("Model loaded!")

# Fashion MNIST class names
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

# Load Fashion MNIST test data
print("Loading test image...")

(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize
x_test = x_test / 255.0

# Select image
index = 10

image = x_test[index]

# Prediction
prediction = model.predict(
    image.reshape(1, 28, 28),
    verbose=0
)

predicted_index = np.argmax(prediction[0])

predicted_class = class_names[predicted_index]

confidence = np.max(prediction[0]) * 100

actual_class = class_names[y_test[index]]

print("\n==============================")
print("FASHION MNIST CLASSIFICATION")
print("==============================")

print("Actual Class    :", actual_class)
print("Predicted Class :", predicted_class)
print("Confidence      :", round(confidence, 2), "%")

print("==============================")