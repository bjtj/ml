#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

int main(int argc, char *argv[])
{
	double x;

	for (x = -3.0; x < 3.0; x += 0.1) {
		printf("%f := %f\n", x, sigmoid(x));
	}
    
    return 0;
}

