# import 
from imutils import paths
import argparse
import imutils
import cv2
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to the input directory of images")
ap.add_argument("-a", "--annot", required=True,
	help="path to the output directory of annotations")
args = vars(ap.parse_args())

# grab the image paths
imagePaths = list(paths.list_images(args["input"]))
counts = {}

print(imagePaths)

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i+1, 
		len(imagePaths)))

	try:
		# load the images, convert it to grayscale,
		# pad the images 
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8,
			cv2.BORDER_REPLICATE)

		# threshold the image to reveal the digits
		thresh = cv2.threshold(gray, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		# find the contours in the image, keep 4 largest ones
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
		
		# loop over the contours
		for c in cnts:
			# compute the bounding box for the contour
			# and extract the digit
			(x, y, w, h) = cv2.boundingRect(c)
			roi = gray[y-5 : y+h+5, x-5 : x+w+5]

			# display the character and capture the keypress
			cv2.imshow("ROI", imutils.resize(roi, width=28))
			key = cv2.waitKey(0)

			# if ' key is pressed, then ignore the character
			if key == ord("'"):
				print("[INFO] ignoring character")
				continue

			# construct a path to the output directory
			key = chr(key).upper()
			dirPath = os.path.sep.join([args["annot"], key])

			# if output directory doea not exist then create it
			if not os.path.exists(dirPath):
				os.mkdirs(dirPath)

			# write the labeled character to file
			count = counts.get(key, 1)
			p = os.path.sep.join([dirPath, "{}.png".format(
				str(count).zfill(6))])
			cv2.imwrite(p, roi)

			# increment the count for current key
			counts[key] += 1

	# manually break from the script
	except KeyboardInterrupt:
		print("[INFO] manually leaving script")
		break

	# some error occured for particular image
	except:
		print("[INFO] skipping image...")

