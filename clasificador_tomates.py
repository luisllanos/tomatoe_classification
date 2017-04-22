# USAGE
# python clasificador_frutas.py --model models/svm.cpickle --video dataset/test_videos/video_example.mov

# python clasificador_tomates.py --model models/svm.cpickle

# import the necessary packages
import cPickle
import mahotas
#from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import cv2
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.rgbhistogram import RGBHistogram
from pyimagesearch import imutils
import mahotas

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-mo", "--model", required = True, help = "path to where the model is stored")
# ap.add_argument("-ti", "--testimages", required = True, help = "path to the image dataset")
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
args = vars(ap.parse_args())

######################## CARGANDO MODELO ###########################
print "Cargando Modelo........."
# load the model
model = open(args["model"]).read()
model = cPickle.loads(model)

########## CARGANDO IMAGENES DEL TRAINING PARA OBTENER LAS ETIQUETAS..... ########
print "Cargando Labels de las imagenes de Entrenamiento..........."
imagenes_entrenamiento = sorted(glob.glob("dataset/images/*.png"))
target = []
# initialize the image descriptor
desc = RGBHistogram([8, 8, 8])
# loop over the image and mask paths
for imagen_entrenamiento in imagenes_entrenamiento:
	# load the image and mask
	image = cv2.imread(imagen_entrenamiento)
	target.append(imagen_entrenamiento.split("_")[-2])

le = LabelEncoder()
target = le.fit_transform(target)

#################################################################################


# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
	camera = cv2.VideoCapture(0)
# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])


print "Clasificando Frutas ........."
# keep looping
while True:
	# grab the current frame
	(grabbed, frame) = camera.read()

	# if we are viewing a video and we did not grab a
	# frame, then we have reached the end of the video
	if args.get("video") and not grabbed:
		print "Sorry, Video Not Found or Bad Frame !"
		break


	#frame = frame.copy()
	# resize the frame and convert it to grayscale
	framed = imutils.resize(frame, width = None, height = 650)
	gray = cv2.cvtColor(framed, cv2.COLOR_BGR2GRAY)
	thresh = gray.copy()
	T = mahotas.thresholding.otsu(thresh)
	thresh[thresh > T] = 255
	thresh[thresh < T] = 0
	thresh = cv2.bitwise_not(thresh)
	features = desc.describe(framed, mask = thresh)
	flower = le.inverse_transform(model.predict(features))[0]
	print "Tomate esta ==> %s" % (flower.upper())
	cv2.putText(framed, "Tomate > {0}".format(str(flower)), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
	cv2.imshow("image", framed)
	


	# if the 'q' key is pressed, stop the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
