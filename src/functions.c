#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double relu(double x) {
	return x > 0.0 ? x : 0.0;
}

typedef double (*func_ptr)(double x);

typedef enum _func_type_e {
	NONE,SIG,TANH,RELU,
} func_type_e;

typedef struct _func_t
{
    func_type_e type;
	func_ptr func;
} func_t;


func_t functions[] = {
	{SIG, sigmoid},
	{TANH, tanh},
	{RELU, relu},
	{NONE, NULL}
};

const char * to_func_type_str(func_type_e type) {
	switch (type) {
	case NONE:
		return "none";
	case SIG:
		return "sigmoid";
	case TANH:
		return "tanh";
	case RELU:
		return "relu";
	default:
		break;
	}
	return "(UNKNOWN)";
}

void list(func_t * func, double from, double to, double step, FILE * out) {
	double x;
	printf("[%s]\n", to_func_type_str(func->type));
	for (x = from; x <= to; x += step) {
		fprintf(out, "%.4f %.4f\n", x, func->func(x));
	}
}

int main(int argc, char *argv[])
{
	double x;

	func_t * elt = functions;

	while (elt->type != NONE) {
		FILE * fp = fopen(to_func_type_str(elt->type), "w");
		if (fp == NULL) {
			printf("Cannot open file\n");
			exit(1);
		}
		list(elt, -3.0, 3.0, 0.1, fp);
		fclose(fp);
		elt++;
	}
    
    return 0;
}

