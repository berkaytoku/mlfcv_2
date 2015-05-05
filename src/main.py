import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '../../caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
PRETRAINED = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
IMAGE_FILE = '../images/IMG_2.jpg'

def getPredictionClassNames():
	classNames = []
	with open(caffe_root + '/data/ilsvrc12/synset_words.txt', 'r') as f:
		for line in f:
			classNames.append(' '.join(line.strip().split(' ')[1:]))
	return classNames

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

input_image = caffe.io.load_image(IMAGE_FILE)
#plt.imshow(input_image)

prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape
#plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()

prediction = net.predict([input_image], oversample=False)
print 'prediction shape:', prediction[0].shape
#plt.plot(prediction[0])
print 'predicted class:', prediction[0].argmax()
print prediction[0][prediction[0].argmax()]

# Resize the image to the standard (256, 256) and oversample net input sized crops.
input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
# 'data' is the input blob name in the model definition, so we preprocess for that input.
caffe_input = np.asarray([net.transformer.preprocess('data', in_) for in_ in input_oversampled])
# forward() takes keyword args for the input blobs with preprocessed input arrays.

caffe.set_mode_gpu()

classNames = getPredictionClassNames()

prediction = net.predict([input_image])
entropy = -1 * np.multiply(prediction[0], np.log2(prediction[0])).sum()
print 'predicted class:', prediction[0].argmax()
print 'prediction probability:', prediction[0][prediction[0].argmax()]
#plt.plot(prediction[0])
print classNames[prediction[0].argmax()]
print entropy
