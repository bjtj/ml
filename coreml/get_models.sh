#!/bin/bash

mkdir -p mtcnn
pushd mtcnn

curl -OL https://github.com/DuinoDu/mtcnn/raw/master/model/det1.caffemodel
curl -OL https://github.com/DuinoDu/mtcnn/raw/master/model/det1.prototxt
curl -OL https://github.com/DuinoDu/mtcnn/raw/master/model/det2.caffemodel
curl -OL https://github.com/DuinoDu/mtcnn/raw/master/model/det2.prototxt
curl -OL https://github.com/DuinoDu/mtcnn/raw/master/model/det3.caffemodel
curl -OL https://github.com/DuinoDu/mtcnn/raw/master/model/det3.prototxt

popd
