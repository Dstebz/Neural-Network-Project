
#include <vector>

double reLU(double ip) {		//compares input with 0. If input, ip, is positive then it returns ip, if negative or 0 then returns 0.
	if (ip <= 0) {
		return 0;
	}
	else {
		return ip;				//returns ip since y=x for non-zero values of x
	}
}

double softmax(std::vector<double> ip) {
	return 0;
}