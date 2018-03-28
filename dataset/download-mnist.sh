#!/bin/bash

TARGET_DIR="."
if [ -n "$1" ]; then
	TARGET_DIR="$1"
fi

URL_LIST="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
URL_LIST+=" http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
URL_LIST+=" http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
URL_LIST+=" http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

if [ ! -d "$TARGET_DIR" ]; then
	mkdir -p "$TARGET_DIR"
fi

for URL in $URL_LIST; do
	FILENAME=$(basename "$URL")
	TARGET_PATH="$TARGET_DIR/$FILENAME"
	echo "[Get '$URL' to '$TARGET_PATH']"
	if [ -f "$TARGET_PATH" ]; then
		echo "[Already exists]"
	else
		curl -L -o "$TARGET_PATH" "$URL"
	fi
done
