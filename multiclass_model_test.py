from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from multiclass_model import fit_mnist_model, evaluate_mnist_model
from multiclass_model import get_mnist_data
from multiclass_model import define_dense_model_single_layer, define_dense_model_with_hidden_layer

def test_define_dense_model_single_layer():
    model = define_dense_model_single_layer(28*28, output_length=10)
    assert len(model.layers) == 1, "Model should have 1 layer"
    assert model.layers[0].input_shape == (None, 28*28), "Input shape is not correct"
    assert model.layers[0].output_shape == (None, 10), "Output shape is not correct"

def test_define_dense_model_with_hidden_layer():
    model = define_dense_model_with_hidden_layer(28*28, hidden_layer_size=50, output_length=10)
    assert len(model.layers) == 2, "Model should have 2 layers"
    assert model.layers[0].input_shape == (None, 28*28), "Input shape is not correct"
    assert model.layers[0].output_shape == (None, 50), "Output shape of hidden layer is not correct"
    assert model.layers[1].output_shape == (None, 10), "Output shape of output layer is not correct"

def test_fit_and_predict_mnist_ten_neurons():
    model = define_dense_model_single_layer(28*28, output_length=10)
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    model = fit_mnist_model(x_train, y_train, model)
    loss, accuracy = evaluate_mnist_model(x_train, y_train, model)
    print("Train Loss:", loss, "Train Accuracy:", accuracy)
    assert accuracy > 0.9, "Accuracy should be greater than 0.9 for training data"
    loss, accuracy = evaluate_mnist_model(x_test, y_test, model)
    print("Test Loss:", loss, "Test Accuracy:", accuracy)
    assert accuracy > 0.9, "Accuracy should be greater than 0.9 for test data"

def test_fit_and_predict_mnist_with_hidden_layers():
    model = define_dense_model_with_hidden_layer(28*28, hidden_layer_size=50, output_length=10)
    (x_train, y_train), (x_test, y_test) = get_mnist_data()
    model = fit_mnist_model(x_train, y_train, model)
    loss, accuracy = evaluate_mnist_model(x_train, y_train, model)
    print("Train Loss:", loss, "Train Accuracy:", accuracy)
    assert accuracy > 0.9, "Accuracy should be greater than 0.9 for training data"
    loss, accuracy = evaluate_mnist_model(x_test, y_test, model)
    print("Test Loss:", loss, "Test Accuracy:", accuracy)
    assert accuracy > 0.9, "Accuracy should be greater than 0.9 for test data"

if __name__ == "__main__":
    test_define_dense_model_single_layer()
    test_define_dense_model_with_hidden_layer()
    test_fit_and_predict_mnist_ten_neurons()
    test_fit_and_predict_mnist_with_hidden_layers()
