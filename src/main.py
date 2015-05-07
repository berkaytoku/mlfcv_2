import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../../caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILES = ['../images/IMG_1.jpg', '../images/IMG_2.jpg', '../images/IMG_3.JPG', '../images/IMG_4.jpg', '../images/IMG_5.jpg', '../images/IMG_6.jpg', '../images/IMG_7.jpg', '../images/IMG_8.jpg']

def getPredictionClassNames():
	classNames = []
	with open(caffe_root + '/data/ilsvrc12/synset_words.txt', 'r') as f:
		for line in f:
			classNames.append(' '.join(line.strip().split(' ')[1:]))
	return classNames

caffe.set_mode_gpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
		       image_dims=(227,227))

classNames = getPredictionClassNames()

for image_name in IMAGE_FILES:
	input_image = caffe.io.load_image(image_name)

	prediction = net.predict([input_image])
	predictedClass = prediction[0].argmax()

	entropy = -1 * np.multiply(prediction[0], np.log2(prediction[0])).sum()
	print 'Image Name: ', image_name
	print 'Predicted Class: ', predictedClass
	print 'Prediction Probability: ', prediction[0][predictedClass]
	print 'Predicted Class Name: ', classNames[predictedClass]
	print 'Prediction Entropy: ', entropy
