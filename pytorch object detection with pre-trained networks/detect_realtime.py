from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2
from pycocotools.coco import COCO

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="frcnn-mobilenet", choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"], help="Name of object detection model")
ap.add_argument("-c", "--confidence", type=float, default=0.9, help="Minimum probability to filter weak detections")

args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# coco data file
dataDir='.'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

COLORS = np.random.uniform(0, 255, size=(91, 3))

# initialize a dictionary containing model name and its corresponding 
# torchvision function call
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True,
	    num_classes=91, pretrained_backbone=True),
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True,
	    num_classes=91, pretrained_backbone=True),
	"retinanet": detection.retinanet_resnet50_fpn(pretrained=True, progress=True,
	    num_classes=91, pretrained_backbone=True)
}

# load the model and set it to evaluation mode
model = MODELS[args["model"]].to(DEVICE)
model.eval()

# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=1280)
    orig = frame.copy()

    # convert the frame from BGR to RGB channel ordering and change
	# the frame from channels last to channels first ordering
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))

    
	# add a batch dimension, scale the raw pixel intensities to the
	# range [0, 1], and convert the frame to a floating point tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.cuda.FloatTensor(frame)

	# send the input to the device and pass the it through the
	# network to get the detections and predictions
    frame.to(DEVICE)
    detections = model(frame)[0]
 
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

    # show the output frame
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
	    break
    
    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()