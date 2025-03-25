import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import os

def load_and_modify_model(model_path, learning_rate=0.001):
    """
    Load a pre-trained model from .h5 file and modify its parameters
    
    Args:
        model_path (str): Path to the .h5 model file
        learning_rate (float): New learning rate for training
    
    Returns:
        model: Modified Keras model
    """
    # Load the pre-trained model
    model = load_model(model_path)
    
    # Modify the model's optimizer with new learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_dataset(data_path, batch_size=32):
    """
    Prepare the dataset for training
    
    Args:
        data_path (str): Path to the dataset directory
        batch_size (int): Batch size for training
    
    Returns:
        train_dataset: TensorFlow dataset for training
        val_dataset: TensorFlow dataset for validation
    """
    # Define image size and other parameters
    img_height = 224  # Adjust based on your model's input size
    img_width = 224
    
    # Create training dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # Create validation dataset
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, val_dataset

def train_model(model, train_dataset, val_dataset, epochs=10, callbacks=None):
    """
    Train the modified model on the new dataset
    
    Args:
        model: Keras model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs (int): Number of training epochs
        callbacks: List of Keras callbacks
    """
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def main():
    # Example usage
    model_path = "path/to/your/model.h5"
    dataset_path = "path/to/your/dataset"
    
    # Load and modify the model
    model = load_and_modify_model(model_path, learning_rate=0.0001)
    
    # Prepare the dataset
    train_dataset, val_dataset = prepare_dataset(dataset_path, batch_size=32)
    
    # Define callbacks (optional)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train the model
    history = train_model(
        model,
        train_dataset,
        val_dataset,
        epochs=10,
        callbacks=callbacks
    )
    
    # Save the trained model
    model.save('trained_model.h5')

if __name__ == "__main__":
    main() 