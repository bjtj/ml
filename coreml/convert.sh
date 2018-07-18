#!/bin/bash

mkdir -p target

for i in $(seq 1 3); do
    CAFFEMODEL_PATH="mtcnn/det$i.caffemodel"
    PROTOTXT_PATH="mtcnn/det$i.prototxt"
    TARGET="target/det$i.mlmodel"
    python convert_caffemodel_to_mlmodel.py $CAFFEMODEL_PATH $PROTOTXT_PATH $TARGET
done
