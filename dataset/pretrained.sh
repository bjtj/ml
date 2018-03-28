#!/bin/bash

# VGG16
# https://github.com/mitmul/chainer-faster-rcnn
if [ ! -d data ]; then mkdir data; fi
curl https://dl.dropboxusercontent.com/u/2498135/faster-rcnn/VGG16_faster_rcnn_final.model?dl=1 -o data/VGG16_faster_rcnn_final.model
