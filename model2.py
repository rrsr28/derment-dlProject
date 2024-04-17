import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall

from sklearn import metrics
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import os
import requests


# --------------------------------------------------------------------------------------------


img_size = (450, 450)

def data_preprocessing(path):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255,
                                                              zoom_range=0.2,
                                                              horizontal_flip=True,
                                                              vertical_flip=True
                                                             )

    generator = datagen.flow_from_directory(
    path,
    batch_size=25,
    class_mode='categorical',
    target_size=img_size,
    color_mode="rgb"
    )

    return generator


train_generator = data_preprocessing("/path/to/train")
validation_generator = data_preprocessing("path/to/validation")


# --------------------------------------------------------------------------------------------


sample_image, _ = next(train_generator)
image_shape = sample_image.shape[1:]
num_channels = image_shape[-1]

print("\nDimensione dell'immagine:", image_shape)
print()
plt.imshow(sample_image[0])
plt.title(f"Dimensione: {image_shape}")
plt.axis('off')
plt.show()


# --------------------------------------------------------------------------------------------


# Define the activation function
act_func = tf.keras.activations.relu

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation=act_func, padding='same', input_shape=(img_size[0], img_size[1], 3), kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation=act_func, padding='same', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(filters=128, kernel_size=5, activation=act_func, padding='same', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation=act_func, padding='same', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation=act_func, padding='same', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=act_func),
    tf.keras.layers.BatchNormalization(axis = -1),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation=act_func),
    tf.keras.layers.BatchNormalization(axis = -1),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(9, activation='softmax')

])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])
history_model = model.fit(train_generator, epochs = 25, validation_data = validation_generator)

model.save("NewModel2.h5")


# --------------------------------------------------------------------------------------------


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predicting labels for the validation set
y_pred = model.predict(validation_generator)
y_true = validation_generator.classes

# Generating the confusion matrix
conf_matrix = confusion_matrix(y_true, np.argmax(y_pred, axis=-1))

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')  # Save the confusion matrix plot
plt.show()

# Plotting accuracy and loss plots
acc = history_model.history['accuracy']
val_acc = history_model.history['val_accuracy']
loss = history_model.history['loss']
val_loss = history_model.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('accuracy_and_loss_plots.png')
plt.show()


# --------------------------------------------------------------------------------------------


class_names = ['Actinic keratosis', 'Atopic Dermatitis', 'Benign keratosis', 'Dermatofibroma', 'Melanocytic nevus', 'Melanoma',  'Squamous cell carcinoma', 'Tinea Ringworm Candidiasis', 'Vascular lesion']

def predict_disease(model, img_path):
  img = tf.keras.utils.load_img(img_path, target_size=(img_size[0], img_size[1], 3), color_mode = 'rgb')
  array = tf.keras.utils.img_to_array(img)
  array = array / 255.0

  img_array = np.expand_dims(array, axis=0)
  preds = model.predict(img_array)

  #formatted_predictions = []
  for prediction in preds:
      formatted_predictions = [f'{value:.2f}' for value in prediction]

  top_prob_index = np.argmax(formatted_predictions)
  top_prob = round(float(formatted_predictions[top_prob_index].replace(",", "."))*100, 2)

  print("Probability for each class:", sorted(zip(class_names, formatted_predictions), key=lambda x: x[1], reverse=True))

  plt.imshow(tf.keras.utils.load_img(img_path, target_size=(img_size[0], img_size[1],3), color_mode = 'rgb'))
  plt.axis('off')
  plt.title(f"Class: {list(class_names)[top_prob_index]}; Prob: {top_prob}%")
  plt.show()

predict_disease(model, "/data/val/Actinic keratosis/ISIC_0025825.jpg")


# --------------------------------------------------------------------------------------------