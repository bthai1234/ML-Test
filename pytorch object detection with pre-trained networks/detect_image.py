from torchvision.models import detection
from pycocotools.coco import COCO
import numpy as np
import argparse
import pickle
import torch
import cv2
from pycocotools.coco import COCO


# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to input image")
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet", choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"], help="Name of object detection model")
ap.add_argument("-c", "--confidence", type=float, default=0.9, help="Minimum probability to filter weak detections")

args = vars(ap.parse_args())

# set Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# coco data file
dataDir='.'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

#load the list of categories in the coco dataset and then generate a set of bounding box colors for each class
COLORS = np.random.uniform(0,255, size=(91,3))

MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True),
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True),
    "retinanet": detection.retinanet_resnet50_fpn(pretrained=True, progress=True, num_classes=91, pretrained_backbone=True)
}

# load the model and set it to evaluation mode
model = MODELS[args["model"]].to(DEVICE)
model.eval()

#load image
image = cv2.imread(args["image"])
orig = image.copy()

#convert the image from BGR to RGB and change the image from channel last to channel first ordering
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2,0,1))

#add the batch dimensions, scale the picels to range [0,1] and convert the image to a floating point tensor
image = np.expand_dims(image, axis=0)
image = image/255.0
image = torch.FloatTensor(image)

# send input to the device and pass it through the network to get the detections and predictions
image = image.to(DEVICE)
detections = model(image)[0]


#loop over the detections
for i in range(0, len(detections["boxes"])):
    #extract the confidence of the predictions
    confidence = detections["scores"][i]

    if confidence > args["confidence"]:
        #extract the index of class labels from detections
        # then compute the (x,y) cord of the bounding box
        idx = int(detections["labels"][i])
        box = detections["boxes"][i].detach().cpu().numpy()
        (startX, startY, endX, endY) = box.astype("int")

        # display the prediction to our terminal
        label = "{}: {:.2f}%".format(coco.loadCats(idx), confidence *100)
        print("[INFO] {}".format(label))

        #draw the bounding box and label on image
        cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY+15
        cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
cv2.imshow("Output", orig)
cv2.waitKey(0)
