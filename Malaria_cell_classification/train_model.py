from difflib import restore
from gc import callbacks
from pickletools import optimize
import matplotlib
matplotlib.use("Agg")

import tensorflow as tf 
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD 
from config import config
from sklearn.metrics import accuracy_score, classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from config import resnet 


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to output loss/accuracy plot")
ap.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs to train")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="batch size")
args = vars(ap.parse_args())

# initial number of epoch learning rate and batch size
EPOCHS = args["epochs"]
LR = 1e-1
BATCH_SIZE = args["batch_size"]

trainTotal = len(list(paths.list_images(config.TRAIN_PATH)))
valTotal = len(list(paths.list_images(config.VAL_PATH)))
testTotal = len(list(paths.list_images(config.TEST_PATH)))

# define function to use for tf's LR schedualer callback version
def poly_decay(epoch):
    # init max num of epochs, base learning rate, and power of poly
    maxEpochs = EPOCHS
    baseLR = LR
    power = 1.0

    # compute new learning rate based on polynomial decay. power of 1 essetually make it a linear decay, increase it to decy LR faster
    alpha = baseLR * (1-(epoch/float(maxEpochs))) ** power

    return alpha

# initialize the training data augmentaion object
trainAug = ImageDataGenerator(
    rescale = 1/255.0,
    rotation_range = 20,
    zoom_range =0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

# initalize theval and test data agmentation to only rescale pixels to [0,1]
valAug = ImageDataGenerator(rescale=1/255.0)

# initalize the training generator
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(64,64),
    color_mode="rgb",
    shuffle=True,
    batch_size=BATCH_SIZE
)    

# initalize the val generator
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(64,64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE
)    

# initalize the val generator
testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(64,64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE
)

# setup earlystopping callback
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoints = tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join("runs", "trains", "{epoch:02d}-{val_loss:.2f}.hdf5"),
    save_best_only=True,
)



# initialize our modified ResNet model and compile it
model = resnet.build(1)
opt = SGD(learning_rate=LR, momentum = 0.9)
model.compile(loss="binary_crossentropy", optimizer= opt, metrics=["accuracy"])

callbacks = [LearningRateScheduler(poly_decay), early_stopping_cb, checkpoints]
history = model.fit(
    x=trainGen,
    steps_per_epoch=trainTotal//BATCH_SIZE,
    validation_data=valGen,
    validation_steps=valTotal//BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks)


# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict(x=testGen, steps=(testTotal // BATCH_SIZE) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))
