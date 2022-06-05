from random import random
from config import config
from imutils import paths
import random
import shutil
import os

# grab path to original data set and shuffle
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# calculate training and testing split index and split into seprate lists 
trainSplit = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:trainSplit]
testPaths = imagePaths[trainSplit:]

# spliting remaing training set for validation
valSplit = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:valSplit]
trainPaths = trainPaths[valSplit:]

# define paths for our splits
datasets = [
    ("training", trainPaths, config.TRAIN_PATH),
    ("validation", valPaths, config.VAL_PATH),
    ("testing", testPaths, config.TEST_PATH)
]

# loop over dataset and copy files to the split directorys
for (dType, imagePaths, splitPath) in datasets:
    # print which data split it creating currently
    print("[INFO] building '{}' split".format(dType))

    # if the output base output directory does not exist, create it
    if not os.path.exists(splitPath):
        print("[INFO] 'creating {}' directory".format(splitPath))
        os.makedirs(splitPath)
    
    # loop over each input image path and copy it into a their respective tarin, val ,or test folder
    for inputPath in imagePaths:
        # exract the filename of the imput image along with its correspinding class label
        filename = inputPath.split(os.path.sep)[-1]
        label = inputPath.split(os.path.sep)[-2]

        # build the path to the label directory 
        labelPath = os.path.sep.join([splitPath, label])

        # if the label output directory does not exist, create it
        if not os.path.exists(labelPath):
            print("[INFO] 'creating {}' directory".format(labelPath))
            os.makedirs(labelPath)

        # construct the path to the destination image and then copy
        # the image itself
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)

