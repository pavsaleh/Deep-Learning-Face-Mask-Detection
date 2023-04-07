# USAGE
# python3 detect_mask_webcam.py

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab dimensions of the frame and construct a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# make a predictions only if at least one face was detected
	if len(faces) > 0:
		# makes predictions on the list of faces
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)

def mask_detection():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str, default="400_model.model", help="path to trained face mask detector model")
	args = vars(ap.parse_args())

	# load our serialized face detector model from disk
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
	weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load the face mask detector model from disk
	print("[INFO] loading face mask detector model...")
	maskNet = load_model(args["model"])
	
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()

	#Information to calculate FPS
	# used to record the time when we processed last frame
	prev_frame_time = 0
	
	# used to record the time at which we processed current frame
	new_frame_time = 0
	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
		frame = vs.read()
		#frame = imutils.resize(frame, width=500)
		
		# time when we finish processing for this frame
		new_frame_time = time.time()
		# detect faces in the frame and determine if they are wearing a face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		# loop over the detected face locations and their corresponding locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			# determine the class label and color we'll use to draw the bounding box and text
			label = "Mask On" if mask > withoutMask else "Mask Off"
			color = (0, 255, 0) if label == "Mask On" else (0, 0, 255)

			# display the label and bounding box rectangle on the output frame
			cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			# Calculating the fps
 
			# fps will be number of frame processed in given time frame
			# since their will be most of time error of 0.001 second
			# we will be subtracting it to get more accurate result
			if new_frame_time - prev_frame_time != 0:
				fps = 1/(new_frame_time-prev_frame_time)
				prev_frame_time = new_frame_time
			else:
				fps = 0
 
			# converting the fps into integer
			fps = int(fps)
		
			# converting the fps to string so that we can display it on frame
			# by using putText function
			fps = str(fps)

			#Print fps
			cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)
		# show the output frame
		cv2.imshow("Face Mask Detector", frame)
		key = cv2.waitKey(1) & 0xFF

if __name__ == "__main__":
	mask_detection()