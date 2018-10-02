#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
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
    if (cap.read(frame)) {
		vector<uchar> buf;
		if (imencode(".jpg", frame, buf) == false) {
			fprintf(stderr, "imencode() failed\n");
			return 1;
		}
		fstream myfile("out.jpg", ios::out | ios::binary);
		myfile.write((char*)&buf[0], buf.size());
		myfile.close();
    }
    return 0;
}
