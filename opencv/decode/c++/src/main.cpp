#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
	if (argc < 2) {
		fprintf(stderr, "usage) %s <image filename>\n", argv[0]);
		return 1;
	}

	ifstream fs(argv[1], ios::in | ios::binary);
	vector<char> buf((istreambuf_iterator<char>(fs)), (istreambuf_iterator<char>()));
	fs.close();

	Mat img = imdecode(InputArray(buf), CV_LOAD_IMAGE_COLOR);
	imshow("preview", img);
	waitKey(0);

    return 0;
}
