# import 
import imutils
import cv2

def preprocess(image, width, height):
	# grab dimensions of the image
	(h, w) = image.shape[:2]

	# if width > height then resize along width
	if w > h:
		image = imutils.resize(image, width=height)
	# otherwise, resize along height
	else:
		image = imutils.resize(image, height=height)

	# determine the padding value
	padW  = int((width - image.shape[1]) / 2.0)
	padH = int((height - image.shape[0]) / 2.0)

	# pad the image and apply one more resizing
	image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
		cv2.BORDER_REPLICATE)
	image = cv2.resize(image, (width, height))

	return image