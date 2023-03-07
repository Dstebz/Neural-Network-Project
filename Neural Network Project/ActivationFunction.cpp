#include "ActivationFunction.h"
#include <vector>
#include <cmath>

double ActivationFunction::reLU(double ip) {	//compares input with 0. If input, ip, is positive then it returns ip, if negative or 0 then returns 0.
	if (ip <= 0) {
		return 0;
	}
	else {
		return ip;				//returns ip since y=x for non-zero values of x
	}
}

double ActivationFunction::tanh(double ip) {
	double ans;

	ans = exp(2 * ip) - 1 / exp(2 * ip) + 1;

	return ans;
}


double ActivationFunction::sigmoid(double ip) {
	double ans;

	ans = 1 / (1 + exp(-ip));

	return ans;
}


double ActivationFunction::softmax(double ip) {
	double ans;

	ans = exp(ip) / 1 + exp(ip);

	return ans;
}
