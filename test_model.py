# import
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from captchahelper import  preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to the input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to input model")
args = vars(ap.parse_args())

# load the network
print("[INFO] loading model...")
model = load_model(args["model"])

# randomly sample a few input images
imagePaths = list(paths.list_images(args["input"]))
imagePaths = np.random.choice(imagePaths, size=(10,),
	replace=False)

# loop over the image 
for imagePath in imagePaths:
	# load the image, convert it to grayscale and pad it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20,
		cv2.BORDER_REPLICATE)

	# threshold the image to get digits
	thresh = cv2.threshold(gray, 0, 255, 
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	# find contours and keep only 4
	cnts = cv2.findContours(thresh.copy(), cv3.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[0]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
	cnts = contours.sort_contours(cnts)[0]

	# init the output image as grayscale with 3 channels
	output = cv2.merge([gray] * 3)
	predictions = []

	# loop over the contours
	for c in cnts:
		# compute the bounding box and extract the digits
		(x, y, w, h) = cv2.boundingRect(c)
		roi = gray[y-5 : y+h+5, x-5 : x+w+5]

		# preprocess the roi and classify it
		roi = preprocess(roi, width=28, height=28)
		roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
		pred = model.predict(roi).argmax(axis=1)[0] + 1
		predictions.append(str(pred))

		# draw prediction on image
		cv2.rectangel(output, (x-2, y-2), (x+w+4, y+h+4), 
			(0, 255, 0), 1)
		cv2.putText(output, str(pred), (x-5, y-5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

	# show the output image
	print("[INFO] captcha: {}".format("".join(predictions)))
	cv2.imshow("Output", output)
	cv2.waitKey()
	