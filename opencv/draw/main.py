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


def draw_rect(img, rect):
    cv2.rectangle(img, (rect.x, rect.y), (rect.r, rect.b), (255, 0, 0), 3)


def main():
    img = load_lena()
    cv2.imshow('preview', img)
    cv2.waitKey(0)

    draw_rect(img, Rect(50, 50, 100, 100))
    cv2.putText(img, 'hello', (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('preview', img)
    cv2.waitKey(0)
    

if __name__ == '__main__':
    main()
