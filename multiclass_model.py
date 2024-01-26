from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_single_layer(input_length, output_length=1):
    """Define a dense model with a single layer for multiclass classification.
    input_length: the number of inputs
    output_length: the number of classes"""
    model = keras.Sequential([
        layers.Dense(output_length, activation='softmax', input_shape=(input_length,))
    ])
    return model

def define_dense_model_with_hidden_layer(input_length, 
                                         hidden_layer_size=10,
                                         output_length=1):
    """Define a dense model with a hidden layer for multiclass classification.
    input_length: the number of inputs
    hidden_layer_size: the number of neurons in the hidden layer
    output_length: the number of classes"""

    model = keras.Sequential([
        layers.Dense(hidden_layer_size, activation='relu', input_shape=(input_length,)),
        layers.Dense(output_length, activation='softmax')
    ])
    return model


def get_mnist_data():
    """Get the MNIST data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255 
    y_train = keras.utils.to_categorical(y_train, 10)  # Convert labels to one-hot encoding
    y_test = keras.utils.to_categorical(y_test, 10)    # Convert labels to one-hot encoding
    return (x_train, y_train), (x_test, y_test)

def fit_mnist_model(x_train, y_train, model, epochs=2, batch_size=32):
    """Fit the model to the data."""
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model
  
def evaluate_mnist_model(x_test, y_test, model):
    """Evaluate the model on the test data."""
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

