#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <ctime>

using namespace cv;

unsigned long _tick() {
    struct timespec tick;
    clock_gettime(CLOCK_MONOTONIC, &tick);
    return tick.tv_sec * 1000 + tick.tv_nsec / 1000000;
}

int main(int argc, char *argv[])
{
    VideoCapture cap(0);
    Mat frame;
    unsigned long tick = _tick();
    int cnt = 0;
    while (cap.read(frame)) {
	imshow("preview", frame);
	if (_tick() - tick >= 1000) {
	    printf("FPS: %d\n", cnt);
	    cnt = 0;
	    tick = _tick();
	}
	cnt++;
	int key = waitKey(1);
	if (key == 'q') {
	    break;
	}
    }
    return 0;
}
