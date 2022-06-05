import os

# original dataset path before pliting
ORIG_INPUT_DATASET = "malaria/cell_images"

# base data set path
BASE_PATH = "malaria"

# Train, val, and test directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define split amount
TRAIN_SPLIT = 0.8

# validation split
VAL_SPLIT = 0.1


