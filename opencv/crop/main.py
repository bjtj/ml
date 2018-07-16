import cv2

class Rect:
    def __init__(self, x, y, w, h):
        "Rectangle"
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.r = x + w
        self.b = y + h


def load_lena():
    return cv2.imread('../lena.jpg')


def crop(img, rect):
    return img[rect.y:rect.b, rect.x:rect.r]


def main():
    img = load_lena()
    cv2.imshow('preview', img)
    cv2.waitKey(0)

    rect = Rect(50, 50, 200, 200)
    cv2.rectangle(img, (rect.x, rect.y), (rect.r, rect.b), (0, 0, 255), 3)
    cv2.imshow('preview', img)
    cv2.waitKey(0)

    cropped = crop(img, rect)
    cv2.imshow('preview', cropped)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
