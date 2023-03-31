#pragma once

#include "pooling.h"
#include <Eigen>					
#include "layer.h"
#include <iostream>

	//don't need activation function for pooling layer due to MAX and AVG pooling methods - no need for weights in this layer



Eigen::MatrixXd PoolingLayer::avg_pool( Eigen::MatrixXd input, int filter, int stride) {
	int ip_width = input.rows();
	int ip_height = input.cols();					//should be square input but have these 2 lines just incase it's rectangular

	int output_rows = (ip_width - filter) / stride + 1;			//working out dimensions of output matrix
	int output_cols = (ip_height - filter) / stride + 1;		//reasoning for formula in logbook i think, +1 used since counting initial instance of filter

	Eigen::MatrixXd output(output_rows, output_cols);			//initialising output matrix

	for (int i = 0; i < output_rows; i++) {						//looping through output matrix		
		for (int j = 0; j < output_cols; j++) {
			int start_row = i * stride;						//updates the starting point for each quadrant is calculated in such a way since it moves n*stride times along hence why it's given by i*stride
			int start_col = j * stride;						//likewise for j*stride
			output(i, j) = input.block(start_row, start_col, filter, filter).mean();		//use of block means that we can section off input matrix into desired quadrants
		}																					//filter used twice in argument since square matrix
	}																						//.mean() used to calculate average of each quadrant
	return output;

}

Eigen::MatrixXd PoolingLayer::max_pool(Eigen::MatrixXd input, int filter, int stride) {			//exact same as avg_pool but use of maxCoeff() instead of mean()
	int ip_width = input.rows();
	int ip_height = input.cols();					

	int output_rows = (ip_width - filter) / stride + 1;			
	int output_cols = (ip_height - filter) / stride + 1;		

	Eigen::MatrixXd output(output_rows, output_cols);			

	for (int i = 0; i < output_rows; i++) {							
		for (int j = 0; j < output_cols; j++) {
			int start_row = i * stride;						
			int start_col = j * stride;						
			output(i, j) = input.block(start_row, start_col, filter, filter).maxCoeff();		//maxCoeff used here to obtain max value of selected quadrant
		}																					
	}																						
	return output;

}


double PoolingLayer::global_pool(Eigen::MatrixXd input, std::string global_pool_type) {
	double ans;					//maybe i should make this case sensitive

	if (global_pool_type == "max") {
		ans = input.maxCoeff();
	}
	else if (global_pool_type == "min") {
		ans = input.minCoeff();
	}
	else if (global_pool_type == "avg") {
		ans = input.mean();
	}
	else {
		std::cout << "invalid pooling type, please enter 'max', 'min' or 'avg'" << std::endl;
	}

	return ans;				//should return a single value
}


PL_Parameters PoolingLayer::getParameters() {
	return this->parameters;
}

void PoolingLayer::setParameters(PL_Parameters params) {
	this->parameters = params;
}


PoolingLayer::PoolingLayer() { //Empty / default constructor

}

PoolingLayer::PoolingLayer(PL_Parameters params) {

}

PoolingLayer::~PoolingLayer() {	//default destructor

}


Eigen::MatrixXd PoolingLayer::Run(Eigen::MatrixXd input) {
	//using max pooling
	Eigen::MatrixXd output;
	output = max_pool(input, this->parameters.filter, this->parameters.stride);

	return output;
}

