
#include "Deconvolution.h"
#include <Eigen>
#include <iostream>

DeconvolutionLayer::DeconvolutionLayer()
{
	this->kernel = Eigen::MatrixXd::Constant(this->parameters.kernelSize, this->parameters.kernelSize, 1);
}; //Empty / default constructor

DeconvolutionLayer::DeconvolutionLayer(DC_Parameters params) {
	this->parameters = params;
	this->kernel = Eigen::MatrixXd::Constant(this->parameters.kernelSize, this->parameters.kernelSize,1);
};

DeconvolutionLayer::DeconvolutionLayer(DC_Parameters params, Eigen::MatrixXd kernel) {
	this->parameters = params;
	this->kernel = kernel;
};

//methods
DC_Parameters DeconvolutionLayer::getParameters() {
	return this->parameters;
}
void DeconvolutionLayer::setParameters(DC_Parameters params) {
	this->parameters = params;
};

Eigen::MatrixXd DeconvolutionLayer::getKernel() {
	return this->kernel;
};

void DeconvolutionLayer::setKernel(Eigen::MatrixXd kernel) {
	this->kernel = kernel;
};


Eigen::MatrixXd DeconvolutionLayer::Run(Eigen::MatrixXd input) { //algorithms based off beyond data science article, see report for citation

	int strideDimensions = (input.rows() - 1) * this->parameters.stride; //dimensions of matrix after stride
	int paddedDimensions = strideDimensions + 2 * this->parameters.padding; //dimensions of matrix after padding
	Eigen::MatrixXd Output = Eigen::MatrixXd::Zero(paddedDimensions, paddedDimensions); //empty output matrix

	for (int i = 0; i < input.rows(); i++) {
		for (int j = 0; j < input.cols(); j++) {
			Output.block(//iterate through input matrix and add to output matrix
				i * this->parameters.stride,//start at 0,0. Increment in multiples of stride
				j * this->parameters.stride,
				this->parameters.kernelSize, //block has same dimensions as kernel
				this->parameters.kernelSize
			) += input(i, j) * this->kernel; //multiply input value by kernel and add to output matrix
		}
	}
	
	return Output; //returns completed output matrix


};

