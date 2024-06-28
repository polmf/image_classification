import os
import csv
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, MaxPooling2D, Flatten
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = '/gpfs/scratch/nct_299/MAMe/MAMe_data_256/data_256'
METADATA_PATH = '/gpfs/scratch/nct_299/MAMe/MAMe_metadata'

np.random.seed(42)
tf.random.set_seed(314)

batch_size = 128
n_epochs = 50

# Mantenim el ImageDataGenerator original
train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATASET_PATH, 'train'),
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42)

val_datagen = ImageDataGenerator(
    rescale=1 / 255.0)

val_generator = val_datagen.flow_from_directory(
    directory=os.path.join(DATASET_PATH, 'val'),
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)

test_datagen = ImageDataGenerator(
    rescale=1 / 255.0)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(DATASET_PATH, 'test'),
    target_size=(256, 256),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)

def load_data_labels(filename):
    labels = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(row[1])
    return labels

def plot_training_curve(history):
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(history.history['loss'], label='train_loss')
    ax[0].plot(history.history['val_loss'], label='val_loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training and Validation Loss')
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(history.history['accuracy'], label='train_acc')
    ax[1].plot(history.history['val_accuracy'], label='val_acc')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training and Validation Accuracy')
    ax[1].grid(True)
    ax[1].legend()
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print('Timestamp curves of this experiment:', timestamp)
    plt.savefig(f'training_curves_{timestamp}.pdf')

def train_cnn():
    img_rows, img_cols, channels = 256, 256, 3
    input_shape = (img_rows, img_cols, channels)
    
    # Construcció del model
    model = Sequential()

    # Capes de convolució i max-pooling amb Batch Normalization
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    # Capes denses i dropout
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(29, activation='softmax'))

    # Reduir la taxa d'aprenentatge per veure si ajuda a la millora
    learning_rate = 0.0001
    adam = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
   
    history = model.fit(train_generator, validation_data=val_generator, epochs=n_epochs, verbose=1, callbacks=[early_stop, reduce_lr])
    
    loss, accuracy = model.evaluate(val_generator, verbose=1)
    print("Validation loss: {:.3f}, Validation accuracy: {:.3f}".format(loss, accuracy))
    
    y_pred = model.predict(val_generator)
    y_true = val_generator.classes
    y_pred = np.argmax(y_pred, axis=1)
    
    labels = load_data_labels(os.path.join(METADATA_PATH, 'MAMe_labels.csv'))
    print(classification_report(y_true, y_pred, target_names=labels))
    print(confusion_matrix(y_true, y_pred))
    
    y_pred_test = model.predict(test_generator)
    y_true_test = test_generator.classes
    y_pred_test = np.argmax(y_pred_test, axis=1)
    print(classification_report(y_true_test, y_pred_test, target_names=labels))
    print(confusion_matrix(y_true_test, y_pred_test))
    
    plot_training_curve(history)

if __name__ == "__main__":
    train_cnn()