#include "ActivationFunction.h"
#include <cmath>
#include <Eigen>



Eigen::MatrixXd reLU(Eigen::MatrixXd ip) {	//reLU method working
	for (int i = 0; i < ip.rows(); i++) {
		for (int j = 0; j < ip.cols(); j++) {					
			if (ip(i,j) <= 0) {
				ip(i, j) = 0;
			}
		}
	}	
	return ip;													
}



Eigen::MatrixXd tanh(Eigen::MatrixXd ip) {			//tanh method working
	Eigen::MatrixXd ans;

	//ans = exp(2 * ip) - 1 / exp(2 * ip) + 1;

	ans = (((2 * ip).array()).exp() - 1) / (((2 * ip).array()).exp() + 1);
		
	return ans;
}



Eigen::MatrixXd sigmoid(Eigen::MatrixXd ip) {		//sigmoid method working 
	Eigen::MatrixXd ans;

	ans = 1 /( 1 + (-ip.array()).exp());

	return ans;
}


Eigen::MatrixXd softmax(Eigen::MatrixXd ip) {		//softmax method working

	double totalExpon = 0;

	for (int i = 0; i < ip.rows(); i++) {
		for (int j = 0; j < ip.cols(); j++) {							//getting sigma part of equation to aid simplicity in next part
			totalExpon += exp(ip(i, j));
		}
	}
	

	for (int i = 0; i < ip.rows(); i++) {
		for (int j = 0; j < ip.cols(); j++) {							//iterating through all elements of matrix
			ip(i, j) = exp(ip(i, j)) / totalExpon;
		}
	}

	return ip;
}

