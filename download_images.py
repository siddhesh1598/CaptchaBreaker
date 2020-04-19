# import
import argparse
import requests
import time
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to the output directory of images")
ap.add_argument("-n", "--num-images", type=int,
	default=500, help="# images to download")
args = vars(ap.parse_args())

# init the URL and total number of images downloaded
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

# loop over the number of images to download
for i in range(args["num_images"]):

	try:
		# try to grab a new captcha image
		r = requests.get(url, timeout=60)

		# save the image to the disk
		p = os.path.sep.join([args["output"], 
			"{}.jpg".format(str(total).zfill(5))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()

		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1

	except:
		print("[INFO] error downloading image...")

	time.sleep(0.1)