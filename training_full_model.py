import keras.losses
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from tensorflow.keras import utils
import models
import helpers
import random
import albumentations as A

# Setup constants/help variables

#path_train = 'E:/DDSM/prom/train_n'
#path_valid = 'E:/DDSM/prom/valid_n'
#path_test = 'E:/DDSM/prom/test_n'

SEED = 69
SHAPE = (128, 128, 1)
TRAIN_EPOCHS = 1000
FILENAME_SUFFIX = "tripletrms"
name_scheme = ["roi_nor", "roi_ben", "roi_mal"]
class_names = ["normal", "cancerous"]
# class_names = ["normal", "benign", "malignant"]
CLASS_COUNT = len(class_names)
RAND_N = 2007

loaded_images = []
labels = []
loaded_nor = []
labels_nor = []



optimizer = optimizers.RMSprop(4.0000e-05)
# optimizer = optimizers.Adam(4.0000e-05)
est = callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=15, restore_best_weights=True)
esc = callbacks.EarlyStopping(monitor="val_output2_accuracy", mode="max", verbose=1, patience=20, restore_best_weights=True)
learn_control = callbacks.ReduceLROnPlateau(monitor='val_output2_accuracy', patience=5, verbose=1,factor=0.2, min_lr=1e-12)

def norm(image, label):
    image = tf.cast(image, tf.float32) / 256.
    return (image, label)

# Dataset and split

print("Preparing the dataset")
count = 0
for scheme in name_scheme:

    print("Loading " + scheme + " cases.")
    for filename in glob.glob("E:\\DDSM\\prom\\base_ddsm_cbm\\"
                              + scheme + "*.png"):
        im = cv2.imread(filename)
        if count == 0:
            labels_nor.append(0)
            loaded_nor.append(im[:, :, 1])
            # labels.append(0)
            # loaded_images.append(im[:, :, 1])
        elif count == 1:
            labels.append(1)
            loaded_images.append(im[:, :, 1])
        elif count == 2:
            labels.append(1)
            loaded_images.append(im[:, :, 1])
    count += 1

loaded_nor = random.choices(loaded_nor, k=RAND_N)
labels_nor = random.choices(labels_nor, k=RAND_N)

loaded_images = [*loaded_images, *loaded_nor]
labels = [*labels, *labels_nor]

loaded_images = np.array(loaded_images).reshape(len(loaded_images),128,128,1)
labels = np.array(labels)

X_train, X_valid, Y_train, Y_valid = train_test_split(loaded_images, labels, test_size=0.30, random_state=SEED)

X_valid, X_test, Y_valid, Y_test = train_test_split(X_valid, Y_valid, test_size=0.5)

transform = A.Compose([
    A.GridDistortion(),
    A.ElasticTransform()
])
X_train_aug = np.zeros(X_train.shape)
for i in range(0,X_train.shape[0]//2):
    tr = transform(image=X_train[i])
    X_train_aug[i] = tr["image"]
for i in range(X_train.shape[0]//2, X_train.shape[0]):
    X_train_aug[i] = X_train[i]

# Prepare & learn - triplet loss

print("Model preparation and training - full model")
# Full model
full_model = models.createFullModel(shape=SHAPE, class_count=CLASS_COUNT)

# Full model
full_model.compile(optimizer=optimizer, loss=[tfa.losses.TripletSemiHardLoss(),keras.losses.SparseCategoricalCrossentropy()],
                   metrics="accuracy", loss_weights=[0.5, 0.25])

# Full model with contrastive
# full_model.compile(optimizer=optimizer, loss=[tfa.losses.ContrastiveLoss(),keras.losses.SparseCategoricalCrossentropy()],
#                    metrics="accuracy", loss_weights=[0.5, 0.25])

full_model.summary()

# Training
stage1 = full_model.fit(X_train_aug, Y_train, batch_size=32, epochs=TRAIN_EPOCHS, validation_data=(X_valid, Y_valid), callbacks=[esc, learn_control])

# Save classes in space to print out

result_test_emb, result_test_class = helpers.save_result(full_model,X_test, Y_test,"test",FILENAME_SUFFIX, class_count=CLASS_COUNT)
result_train_emb, result_train_class = helpers.save_result(full_model,X_train, Y_train,"train",FILENAME_SUFFIX, class_count=CLASS_COUNT)

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(result_train_emb, Y_train)
y_test_pred_knn = knn.predict_proba(result_test_emb)

y_test_pred_c = (result_test_class + y_test_pred_knn) / 2.0

helpers.save_results_knn(Y_test, y_test_pred_c, CLASS_COUNT, FILENAME_SUFFIX)

# Summarize history for accuracy classification

plt.plot(stage1.history['output2_accuracy'])
plt.plot(stage1.history['val_output2_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('./PLOTS/' + FILENAME_SUFFIX + '_accuracy')
plt.show()

# Summarize history for loss classification

plt.plot(stage1.history['output2_loss'])
plt.plot(stage1.history['val_output2_loss'])
plt.title('model loss - classification')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('./PLOTS/' + FILENAME_SUFFIX + '_classloss')
plt.show()

# Summarize history for loss embeddings

plt.plot(stage1.history['output1_loss'])
plt.plot(stage1.history['val_output1_loss'])
plt.title('model loss - embeddings')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.savefig('./PLOTS/' + FILENAME_SUFFIX + '_embloss')
plt.show()

# Save weights

full_model.save_weights("./MODELS/model_values_"+FILENAME_SUFFIX+".h5")