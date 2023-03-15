#pragma once

#include "pooling.h"
#include <Eigen>					
#include "layer.h"

	//don't need activation function for pooling layer due to MAX and AVG pooling methods - no need for weights in this layer

Eigen::MatrixXd max_pool( Eigen::MatrixXd input, int filter, int stride) {
	Eigen::MatrixXd ans;
	

	
	
	return ans;
}

Eigen::MatrixXd avg_pool( Eigen::MatrixXd input, int filter, int stride) {
	Eigen::MatrixXd ans;
	double total = 0.0;


	int count;
	double average = 0.0;
		
	for (int i = 0; i < filter; i++) {
		for (int j = 0; j < filter; i++) {
			total += input(i, j);
			count++;
		}
	}
	average = total / count;

	return ans;
}


Eigen::MatrixXd global_pool( Eigen::MatrixXd input ) {
	Eigen::MatrixXd ans;




	return ans;				//should return a 1x1 matrix
}