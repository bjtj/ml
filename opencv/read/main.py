import os
import cv2
import numpy as np


def main():
    with open('../lena.jpg', 'rb') as f:
        data = f.read()
        img = cv2.imdecode(np.fromstring(data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('title', img)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()
