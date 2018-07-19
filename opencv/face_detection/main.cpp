// https://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
// https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/objectDetection/objectDetection.cpp
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <ctime>

using namespace std;
using namespace cv;

unsigned long _tick() {
    struct timespec tick;
    clock_gettime(CLOCK_MONOTONIC, &tick);
    return tick.tv_sec * 1000 + tick.tv_nsec / 1000000;
}

void detectAndDisplay(Mat frame);

String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

int main(int argc, char *argv[])
{

    if (!face_cascade.load(face_cascade_name)){
	cerr << "--(!)Error loading" << endl;
	return -1;
    }
    
    if (!eyes_cascade.load(eyes_cascade_name)){
	cerr << "--(!)Error loading" << endl;
	return -1;
    }
    
    VideoCapture cap(0);
    Mat frame;
    unsigned long tick = _tick();
    int cnt = 0;
    
    while (cap.read(frame)) {
	detectAndDisplay(frame);
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

void detectAndDisplay(Mat frame) {
    vector<Rect> faces;
    Mat frame_gray;
    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);
    face_cascade.detectMultiScale(frame_gray, faces);

    for (size_t i = 0; i < faces.size(); ++i) {
	Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
	ellipse(frame, center, Size(faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar(255, 0, 255), 4);
	Mat faceROI = frame_gray(faces[i]);

	vector<Rect> eyes;
	eyes_cascade.detectMultiScale(faceROI, eyes);

	for (size_t j = 0; j < eyes.size(); ++j) {
	    Point eye_center(faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2);
	    int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
	    circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
	}

    }
    imshow("preview", frame);
}
