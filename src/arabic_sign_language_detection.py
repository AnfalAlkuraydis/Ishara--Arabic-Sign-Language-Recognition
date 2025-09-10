# library imports
import zipfile
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from PIL import ImageFile
import random
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

# Unzips the Arabic Sign Language dataset
zip_ref = zipfile.ZipFile('/content/rgb-arabic-alphabets-sign-language-dataset.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

dataset_path = "/content/RGB ArSL dataset"

# Checks if a given directory exists, lists its contents, and labels each item as a file or directory
def check_dataset_contents(directory):
    print(f"Checking contents of the directory: {directory}")

    if not os.path.exists(directory):
        print("Directory does not exist.")
        return

    contents = os.listdir(directory)

    if not contents:
        print("The directory is empty.")
        return

    print("Contents of the directory:")
    for item in contents:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            print(f"Directory: {item}")
        else:
            print(f"File: {item}")

# Loads images from subdirectories, resizes them to 128x128, normalizes pixel values, 
# and returns arrays of images and their corresponding labels (folder names).
check_dataset_contents(dataset_path)

def load_and_preprocess_images(directory):
    images = []
    labels = []

    for category in os.listdir(directory):
        category_path = os.path.join(directory, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    try:
                        img = load_img(img_path, target_size=(128, 128))
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(category)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)


ImageFile.LOAD_TRUNCATED_IMAGES = True


# Verifies integrity of all images in a folder and deletes any corrupted or unreadable files
def verify_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                img = Image.open(img_path)  # Attempt to open the image
                img.verify()                # Verify the integrity of the image
            except (IOError, SyntaxError) as e:
                print(f"Corrupted image deleted: {img_path}")
                os.remove(img_path)  # Delete the corrupted image

# Use the function to check for corrupted images in the dataset folder
verify_images('/content/RGB ArSL dataset')

images, labels = load_and_preprocess_images(dataset_path)

# Encodes string labels into numeric values, converts them to categorical format, 
# and prints dataset statistics (number of images, shape, and unique labels).
label_encoder = LabelEncoder()
encoded_label = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_label)
if len(images) > 0:
    print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]} and labels: {np.unique(labels)}")
else:
    print("No images found. Please check the dataset path.")

# Splits the dataset into training, validation, and test sets (80/10/10), 
# converts them to NumPy arrays, and prints the shapes of each set.
X_train, X_test, y_train, y_test = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)
X_test,X_val,y_test,y_val=train_test_split(X_test,y_test,test_size=0.5, random_state=42)
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)
print(f'X_train shape is {X_train.shape}')
print(f'X_val shape is {X_val.shape}')
print(f'X_test shape is {X_test.shape}')
print(f'y_train shape is {y_train.shape}')
print(f'y_val shape is {y_val.shape}')
print(f'y_test shape is {y_test.shape}')


# Builds a transfer learning model using MobileNet as the base (without top layers),
# adds custom dense layers for classification, and sets the final output layer 
# to predict 31 classes with softmax activation.

input_shape = (128, 128, 3)
num_classes = 31

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compiles the model with Adam optimizer and categorical crossentropy loss, 
# and sets accuracy as the evaluation metric. Also defines early stopping 
# to prevent overfitting by monitoring validation loss with a patience of 5 epochs.

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Builds the model with the specified input shape and prints a summary of its architecture
model.build(input_shape=(None, 128, 128, 3))
print(model.summary())

# Trains the model on the training set with validation, using early stopping to prevent overfitting,
# and stores the training history (loss/accuracy per epoch) for later analysis.
history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=35,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model on the test set and print loss and accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# Training plot and validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot( 1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title ('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

#plot training  and loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()


model.save('arabic_sign_language_model.h5')
files.download('arabic_sign_language_model.h5')
