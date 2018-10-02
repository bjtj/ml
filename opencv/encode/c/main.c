#include <unistd.h>
#include <stdio.h>
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"

int main(int argc, char *argv[])
{

	CvCapture * cap = cvCaptureFromCAM(0);
	IplImage * frame = cvQueryFrame(cap);
	CvMat * mat = cvEncodeImage(".jpg", frame, 0);

	printf("mat -- %p, rows: %d, cols: %d, step: %d, type: %d\n", mat->data.ptr, mat->rows, mat->cols, mat->step, mat->type);

	FILE * fp = fopen("out.jpg", "wb");
	if (fp == NULL) {
		fprintf(stderr, "fopend() failed");
		return 1;
	}
	fwrite(mat->data.ptr, 1, mat->step, fp);
	fclose(fp);
    
    return 0;
}
