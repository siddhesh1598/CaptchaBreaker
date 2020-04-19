# import
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import SGD
from lenet import LeNet
from captchahelper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the input dataset")
ap.add_argument("-o", "--output", required=True,
	help="path to output model")
args = vars(ap.parse_args())

# init data and labels
data = []
labels = []

# loop over the images in the dataset
for imagePath in paths.list_images(args["dataset"]):
	# load the image, preprocess it and store it in data list
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = preprocess(image, width=28, height=28)
	image = img_to_array(image)
	data.append(image)

	# extract label from the image path and append to labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# normalize the data
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split the data into train/test
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25)

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.transform(trainY)
testY = lb.transform(testY)

# init the model
print("[INFO] compiling model...")	
model = LeNet.build(width=28, height=28, depth=1, 
	classes=9)
opt = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=15, batch_size=32, verbose=1)

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=lb.classes_))

# save the model
print("[INFO] saving the model...")
model.save(args["model"])

# plot the training loss and acc
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("# Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
