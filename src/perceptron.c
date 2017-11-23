#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DEBUG 0

#define LEARNING_RATE 0.1
#define MAX_DATA_SET 500
#define MAX_ITERATION 100

int activate(float wx, float wy, float wb, float x, float y) {
	return ((x * wx + y * wy + wb) >= 0) ? 1 : -1;
}

void randomWeights(float * x, float * y, float * b) {
	*x = (float)rand() / (float)RAND_MAX;
	*y = (float)rand() / (float)RAND_MAX;
	*b = (float)rand() / (float)RAND_MAX;
}

int main(int argc, char *argv[])
{
	const char * path = "train.txt";
	FILE * fp = NULL;
	float wx, wy, wb;
	float x[MAX_DATA_SET], y[MAX_DATA_SET];
	int answer[MAX_DATA_SET];
	int i, count;
	float tx, ty;

	srand(time(NULL));

	if (argc >= 2) {
		path = argv[1];
	}

	fp = fopen(path, "r");
	if (fp == NULL) {
		printf("Cannot open file '%s'", path);
		exit(1);
	}

	randomWeights(&wx, &wy, &wb);

	for (i = 0; i < MAX_DATA_SET; ++i) {
		if (fscanf(fp, "%f %f %d", &x[i], &y[i], &answer[i]) == EOF) {
			break;
		}
#if DEBUG
		printf("%.4f %.4f => %d\n", x[i], y[i], answer[i]);
#endif
	}

	printf("[TRAIN]\n");

	count = i;

	for (i = 0; i < MAX_ITERATION; i++) {
		int j = 0;
		float global_error = 0;
		for (j = 0; j < count; ++j) {
			int predict = activate(wx, wy, wb, x[j], y[j]);
			float error = answer[j] - predict;
			wx += LEARNING_RATE * error * x[j];
			wy += LEARNING_RATE * error * y[j];
			wb += LEARNING_RATE * error;

#if DEBUG
			printf("%.4f * %.4f + %.4f * %.4f + %.4f => %d [%d] (Error: %.4f)\n",
				   x[j], wx, y[j], wy, wb, predict, answer[j], error);
#endif

			global_error += (error * error);
		}
		printf("Iteration %d : RMSE = %.4f\n", i, sqrt(global_error / count));
		if (global_error == 0) {
			break;
		}
	}

	printf("[DONE]\n");

	printf("[TEST]\n");

	while (1) {
		printf("Enter 'x y' (Ctrl+D to quit): ");
		if (fscanf(stdin, "%f %f", &tx, &ty) == EOF) {
			break;
		}
		printf("%.4f * %.4f + %.4f * %.4f + %.4f = %d\n", tx, wx, ty, wy, wb, activate(wx, wy, wb, tx, ty));
	}
	
	printf("\n[BYE]\n");
	
    return 0;
}

