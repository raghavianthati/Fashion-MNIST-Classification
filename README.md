# Fashion MNIST Classification

## Project Overview

Fashion MNIST Classification is a Deep Learning project that identifies different types of clothing from grayscale images.

The model is trained using the Fashion MNIST dataset and predicts one of 10 clothing categories.

## Categories

The model can classify:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Dataset

The Fashion MNIST dataset contains 70,000 grayscale images.

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- 10 classes

## Model

The project uses a Neural Network with:

1. Flatten Layer
2. Dense Layer with ReLU
3. Output Layer with Softmax

## Project Structure

Fashion_MNIST_Classification/

├── train.py  
├── predict.py  
├── fashion_model.keras  
├── README.md  
└── requirements.txt

## How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt