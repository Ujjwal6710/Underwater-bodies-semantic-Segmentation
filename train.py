#!/usr/bin/env python3

import argparse
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
from glob import glob
from utils import ShowProgress, save_model
from models import Unet, FCN8, DeepLab, DataGenerator, iou_loss, dice_loss
import gc
import tensorflow.keras.backend as K
import warnings
warnings.filterwarnings("ignore")

gc.collect()
K.clear_session()

MODELS = {
    "unet": Unet(),
    "fcn": FCN8(),
    "deeplab": DeepLab()
}

gpus = tf.config.list_physical_devices('GPU')

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--test_data_path", type=str)
    args = parser.parse_args()

    # Training Data paths
    train_images = sorted(glob(f"{args.data_path}/images/*.jpg"))
    train_masks = sorted(glob(f"{args.data_path}/masks/*.bmp"))

    # Testing Data paths
    test_images = sorted(glob(f"{args.test_data_path}/images/*.jpg"))
    test_masks = sorted(glob(f"{args.test_data_path}/masks/*.bmp"))

    # Check if training images and masks are found
    if not train_images or not train_masks:
        raise FileNotFoundError("No training images or masks found. Please check the data path and file structure.")
    
    # Check if testing images and masks are found
    if not test_images or not test_masks:
        raise FileNotFoundError("No testing images or masks found. Please check the test data path and file structure.")

    train_data = [(image, mask) for image, mask in zip(train_images, train_masks)]
    test_data = [(image, mask) for image, mask in zip(test_images, test_masks)]
    
    # Create Data Generators
    train_generator = DataGenerator(train_data)
    test_generator = DataGenerator(test_data)

    # Create model
    model = MODELS[args.model]

    # Compile the model
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["accuracy"]
    )

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=args.epochs,
        callbacks=[ShowProgress(train_generator)],
        batch_size=8
    )

    # Save the model
    save_model(model, save_path=f"saved_models/{args.model}.pkl")
    
    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    
    plt.savefig(f"graph{args.model}.jpg")
    plt.show()

