#!/usr/bin/env python3
# CNN Practical Demonstration
# This script demonstrates the implementation of Convolutional Neural Networks
# with practical examples using TensorFlow/Keras

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
import os

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create a directory for saving visualizations
os.makedirs('images', exist_ok=True)

# Function to plot images
def plot_images(images, labels, class_names, filename=None):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

# Function to plot model training history
def plot_training_history(history, filename=None):
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

# Function to visualize feature maps
def visualize_feature_maps(model, image, layer_name, filename=None):
    # Ensures the image has the correct shape: (1, 32, 32, 3)
    image = np.expand_dims(image, axis=0)

    layer = model.get_layer(name=layer_name)

    # Get the feature maps for our input image
    activation_model = tf.keras.models.Model(
        inputs=model.layers[0].input,  # pega a entrada da primeira camada
        outputs=layer.output
    )

    # Generates the activations
    feature_maps = activation_model.predict(image)

    # Plot up to 16 feature maps or all if less than 16
    plt.figure(figsize=(15, 8))
    num_features = min(16, feature_maps.shape[-1])
    for i in range(num_features):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.axis('off')

    plt.suptitle(f'Mapas de Características da camada {layer_name}')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


# Function to visualize filters
def visualize_filters(model, layer_name, filename=None):
    # Get the layer
    for layer in model.layers:
        if layer.name == layer_name:
            filters, biases = layer.get_weights()
            break
    
    # Normalize filter values to 0-1
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    # Plot the filters
    plt.figure(figsize=(15, 8))
    
    # Plot up to 16 filters or all if less than 16
    num_filters = min(16, filters.shape[-1])
    for i in range(num_filters):
        plt.subplot(4, 4, i + 1)
        # For RGB images, we'll visualize the filter for the first channel
        plt.imshow(filters[:, :, 0, i], cmap='viridis')
        plt.xticks([])
        plt.yticks([])
    
    plt.suptitle(f'Filters from {layer_name}')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

# Function to create a simple CNN model for CIFAR-10
def create_simple_cnn():
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), name='conv1'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool3'),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax', name='output')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Function to create a LeNet-5 inspired model
def create_lenet_model():
    model = models.Sequential([
        # First Convolutional Layer
        layers.Conv2D(6, (5, 5), activation='relu', padding='same', input_shape=(32, 32, 3), name='conv1_lenet'),
        layers.AveragePooling2D((2, 2), name='pool1_lenet'),
        
        # Second Convolutional Layer
        layers.Conv2D(16, (5, 5), activation='relu', name='conv2_lenet'),
        layers.AveragePooling2D((2, 2), name='pool2_lenet'),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(120, activation='relu', name='dense1_lenet'),
        layers.Dense(84, activation='relu', name='dense2_lenet'),
        layers.Dense(10, activation='softmax', name='output_lenet')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Main function to run the demonstration
def main():
    print("Loading CIFAR-10 dataset...")
    # Load and prepare the CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    train_labels_one_hot = to_categorical(train_labels, 10)
    test_labels_one_hot = to_categorical(test_labels, 10)
    
    # Define class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Plot some sample images
    print("Plotting sample images...")
    plot_images(train_images[:25], train_labels.reshape(-1), class_names, 
                filename='images/cifar10_samples.png')
    
    # Create and train a simple CNN model
    print("Creating and training a simple CNN model...")
    simple_model = create_simple_cnn()
    simple_model.summary()
    
    # Train the model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = simple_model.fit(
        train_images, train_labels_one_hot,
        epochs=20,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history, filename='images/simple_cnn_training_history.png')
    
    # Evaluate the model
    print("Evaluating the model...")
    test_loss, test_acc = simple_model.evaluate(test_images, test_labels_one_hot, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Visualize feature maps
    print("Visualizing feature maps...")
    sample_image = test_images[0]
    visualize_feature_maps(simple_model, sample_image, 'conv1', 
                          filename='images/feature_maps_conv1.png')
    visualize_feature_maps(simple_model, sample_image, 'conv2', 
                          filename='images/feature_maps_conv2.png')
    
    # Visualize filters
    print("Visualizing filters...")
    visualize_filters(simple_model, 'conv1', filename='images/filters_conv1.png')
    
    # Create and train a LeNet-5 inspired model for comparison
    print("Creating and training a LeNet-5 inspired model...")
    lenet_model = create_lenet_model()
    lenet_model.summary()
    
    lenet_history = lenet_model.fit(
        train_images, train_labels_one_hot,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot LeNet training history
    print("Plotting LeNet training history...")
    plot_training_history(lenet_history, filename='images/lenet_training_history.png')
    
    # Evaluate the LeNet model
    print("Evaluating the LeNet model...")
    lenet_test_loss, lenet_test_acc = lenet_model.evaluate(test_images, test_labels_one_hot, verbose=2)
    print(f"LeNet Test accuracy: {lenet_test_acc:.4f}")
    
    # Compare model predictions
    print("Comparing model predictions...")
    # Get predictions from both models
    simple_predictions = simple_model.predict(test_images[:10])
    lenet_predictions = lenet_model.predict(test_images[:10])
    
    # Convert predictions to class indices
    simple_pred_classes = np.argmax(simple_predictions, axis=1)
    lenet_pred_classes = np.argmax(lenet_predictions, axis=1)
    true_classes = test_labels[:10].reshape(-1)
    
    # Display the results
    plt.figure(figsize=(15, 10))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i])
        plt.title(f"True: {class_names[true_classes[i]]}\n"
                 f"Simple: {class_names[simple_pred_classes[i]]}\n"
                 f"LeNet: {class_names[lenet_pred_classes[i]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/model_comparison.png')
    plt.show()
    
    print("Demonstration completed successfully!")

if __name__ == "__main__":
    main()
