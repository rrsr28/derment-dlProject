import os
import cv2
import keras
import pylab as pl
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras.applications import VGG19,EfficientNetB0,VGG16,InceptionV3,ResNet50,EfficientNetB3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten, Dense

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from mlxtend.plotting import plot_confusion_matrix

from keras.applications.vgg16 import preprocess_input

import warnings
warnings.filterwarnings("ignore")


# -----------------------------------

# Define your categories
cat = ['Acne and Rosacea Photos',
       'Normal',
       'vitiligo',
       'Tinea Ringworm Candidiasis and other Fungal Infections',
       'Melanoma Skin Cancer Nevi and Moles',
       'Eczema Photos']

cat2 = ['Melanoma Skin Cancer Nevi and Moles']


def data_dictionary():
    # Adjust the path to your local directory
    path_train = r"C:/Users/rrsan/Documents/My Docs/College/Projects/dermnet-dlProject/Dataset/skin/train"
    path_test = r"C:/Users/rrsan/Documents/My Docs/College/Projects/dermnet-dlProject/Dataset/dermnet/train"

    # Initialize dictionaries to store image paths and labels
    train_dictionary = {"image_path": [], "target": []}

    # Loop through categories in train directory
    k = 0
    for i in cat:
        path_disease_train = os.path.join(path_train, i)
        image_list_train = os.listdir(path_disease_train)
        for j in image_list_train:
            img_path_train = os.path.join(path_disease_train, j)
            # Exclude specific image path if needed
            if img_path_train != r"C:/Users/rrsan/Documents/My Docs/College/Projects/dermnet-dlProject/Dataset/skin/train/Normal\34.avif":
                train_dictionary["image_path"].append(img_path_train)
                train_dictionary['target'].append(k)
        k += 1

    # Loop through categories in test directory
    for i in cat2:
        path_disease_test = os.path.join(path_test, i)
        image_list_test = os.listdir(path_disease_test)
        for j in image_list_test:
            img_path_train = os.path.join(path_disease_test, j)
            train_dictionary["image_path"].append(img_path_train)
            train_dictionary['target'].append(4)

    # Create DataFrame
    train_df = pd.DataFrame(train_dictionary)
    return train_df


# Call the function to get the dataframe
train = data_dictionary()

# -----------------------------------

images=[]
label=[]
for i in train['image_path']:
    img=cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(180,180))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img=cv2.resize(img,(64,64),interpolation=cv2.INTER_AREA)
    #img=resize_and_rescale(img)
    images.append(img)

# -----------------------------------


"""# Create a list of example inputs to our Gradio demo
example_list = [image for image in random.sample(images, k=6)]

# Plot the images in a 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

for ax, img_array in zip(axes.ravel(), example_list):
    ax.imshow(img_array)
    ax.axis('off')

plt.show()"""


# -----------------------------------

label=train['target']
data=np.array(images)
label=np.array(label)
print(label)

vgg_model = VGG19(weights='imagenet',  include_top = False, input_shape=(180, 180, 3))

# let's make all layers non-trainable
for layer in vgg_model.layers :
    layer.trainable = False

# Display the model summary
vgg_model.summary()

# -----------------------------------

model = Sequential([

    Dense(200, activation='relu'),
    Dense(170, activation='relu'),
    Dense(6, activation='softmax'),

])

mcp_save = ModelCheckpoint('EnetB0_CIFAR10_TL.keras', save_best_only=True, monitor='val_acc')
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, verbose=1,)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

kf = KFold(n_splits = 3)

acc = []
num_classes = 6
label = keras.utils.to_categorical(label, num_classes)

# -----------------------------------


x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state = np.random.randint(1,1000, 1)[0])
print(x_train.shape)

# let's make all layers non-trainable
for layer in vgg_model.layers :
    layer.trainable = False

features_train=vgg_model.predict(x_train)
features_test=vgg_model.predict(x_test)
print(features_train.shape)
num_train=x_train.shape[0]
num_test=x_test.shape[0]
print(num_train)
print(num_test)
x_test=features_test.reshape(num_test,-1)
x_train=features_train.reshape(num_train,-1)
print(x_train.shape)
print(x_train.shape)

history = model.fit(x_train, y_train, epochs=25)
model.save('6Classes.h5')

y_pred=model.predict(x_test)
y_pred2=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)

confusion_matrix_result=confusion_matrix(y_test,y_pred2)

plt.title("Confusion_matrix")
ax= plt.subplot()
sns.heatmap(confusion_matrix_result, annot=True, fmt='g', ax=ax);
plt.savefig('Confuse.png')
pl.show()

train_accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(len(train_accuracy))
plt.figure(figsize=(12,4))
# Plotting the accuracy
plt.subplot(1,2,1)
plt.plot(epochs, train_accuracy, 'b', label='Training accuracy')

plt.title('accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'val'], loc='lower right')

# Plotting the loss
plt.subplot(1,2,2)
plt.plot(epochs, train_loss, 'b', label='Training loss')

plt.title('Loss ')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('training_plots.png')
plt.show()


confusion_matrix = classification_report(y_test,y_pred2)
print(confusion_matrix)


def predict_skin_disease(image_path):
    # Load saved model
    model = tf.keras.models.load_model(r"C:/Users/rrsan/Documents/My Docs/College/Projects/dermnet-dlProject/models/6Classes.h5")

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = vgg_model.predict(img)
    img = img.reshape(1, -1)

    # Make prediction on preprocessed image
    pred = model.predict(img)[0]
    predicted_class = np.argmax(pred)

    return predicted_class


print(predict_skin_disease(r"C:/Users/rrsan/Documents/My Docs/College/Projects/dermnet-dlProject/Dataset/skin/test/Normal/0_0_aidai_0029.jpg"))

def predict_skin_disease(image_path):
    # Define list of class names
    class_names = ["Acne","Eczema","Atopic","Psoriasis","Tinea","vitiligo"]

    # Load saved model
    model = tf.keras.models.load_model(r"C:/Users/rrsan/Documents/My Docs/College/Projects/dermnet-dlProject/models/6Classes.h5")

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    img = vgg_model.predict(img)
    img = img.reshape(1, -1)

    # Make prediction on preprocessed image
    pred = model.predict(img)[0]
    predicted_class_index = np.argmax(pred)
    predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

print(predict_skin_disease(r"C:/Users/rrsan/Documents/My Docs/College/Projects/dermnet-dlProject/Dataset/dermnet/test/Atopic Dermatitis Photos/03ichthyosis050127.jpg"))