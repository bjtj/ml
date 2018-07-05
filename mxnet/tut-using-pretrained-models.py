import json
import os
import sys
import mxnet as mx
import matplotlib.pyplot as plt
from mxnet import gluon, nd
from mxnet.gluon.model_zoo import vision
import numpy as np
import cv2


ctx = mx.cpu()
densenet121 = vision.densenet121(pretrained=True, ctx=ctx)
mobileNet = vision.mobilenet0_5(pretrained=True, ctx=ctx)
resnet18 = vision.resnet18_v1(pretrained=True, ctx=ctx)

print(mobileNet)
print(mobileNet.features[0].params)
print(mobileNet.output)

mx.test_utils.download('https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/onnx/image_net_labels.json')
categories = np.array(json.load(open('image_net_labels.json', 'r')))
print(categories[4])


# get a test image
filename = mx.test_utils.download('https://github.com/dmlc/web-data/blob/master/mxnet/doc/tutorials/onnx/images/dog.jpg?raw=true', fname='dog.jpg')

# load the image as a ndarray
image = mx.image.imread(filename)
# plt.imshow(image.asnumpy())
cv2.imshow('preview', cv2.cvtColor(image.asnumpy(), cv2.COLOR_RGB2BGR))

cv2.waitKey(0)

def transform(image):
    resized = mx.image.resize_short(image, 224)
    cropped, crop_info = mx.image.center_crop(resized, (224, 224))
    normalized = mx.image.color_normalize(cropped.astype(np.float32) / 255,
                                          mean=mx.nd.array([0.485, 0.456, 0.406]),
                                          std=mx.nd.array([0.229, 0.224, 0.225]))

    transposed = normalized.transpose((2,0,1)) # transposing from (224, 224, 3) to (3, 224, 224)
    batchified = transposed.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    return batchified


predictions = resnet18(transform(image)).softmax()
print(predictions.shape)

top_pred = predictions.topk(k=3)[0].asnumpy()

for index in top_pred:
    probability = predictions[0][int(index)]
    category = categories[int(index)]
    print("{}: {:.2f}%".format(category, probability.asscalar()*100))


def predict(model, image, categories, k):
    predictions = model(transform(image)).softmax()
    top_pred = predictions.topk(k=k)[0].asnumpy()
    for index in top_pred:
        probability = predictions[0][int(index)]
        category = categories[int(index)]
        print("{}: {:.2f}%".format(category, probability.asscalar()*100))
    print('')


predict(densenet121, image, categories, 3)


predict(mobileNet, image, categories, 3)


predict(resnet18, image, categories, 3)


# fine tuning
NUM_CLASSES=10
with resnet18.name_scope():
    resnet18.output = gluon.nn.Dense(NUM_CLASSES)

print(resnet18.output)
