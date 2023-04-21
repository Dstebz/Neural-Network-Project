
//#include "ActivationFunction.h"
#include "Convolution.h"
#include <Eigen>

ConvolutionLayer::~ConvolutionLayer() {
	//Destructor
}

ConvolutionLayer::ConvolutionLayer() { //Empty / default constructor
	this->kernel = Eigen::MatrixXd::Constant(this->parameters.kernelSize, this->parameters.kernelSize,1); //square Matrix, initialised with 1s
}

ConvolutionLayer::ConvolutionLayer(C_Parameters params) {
	this->parameters = params;
	this->kernel = Eigen::MatrixXd::Constant(this->parameters.kernelSize, this->parameters.kernelSize, 1); //square Matrix, initialised with 1s
}



//methods
C_Parameters ConvolutionLayer::getParameters() {
	return this->parameters;
}
void ConvolutionLayer::setParameters(C_Parameters params) {
	this->parameters = params;
}
Eigen::MatrixXd ConvolutionLayer::getKernel() {
	return this->kernel;
}
void ConvolutionLayer::setKernel(Eigen::MatrixXd kernel) {
	this->kernel = kernel;
}

Eigen::MatrixXd ConvolutionLayer::Run(Eigen::MatrixXd input) {
	int outputDimensionX = (input.rows() -  2 * this->parameters.padding) / this->parameters.stride;
	int outputDimensionY = (input.cols() - 2 * this->parameters.padding) / this->parameters.stride;
	
	Eigen::MatrixXd output = Eigen::MatrixXd::Zero(outputDimensionX, outputDimensionY);

	for (int i = 0; i < outputDimensionX; i++) {
		for (int j = 0; j < outputDimensionY; j++) {
			output(i, j) = (input.block(i * this->parameters.stride, j * this->parameters.stride, this->parameters.kernelSize, this->parameters.kernelSize) //start at i,j, take kernelSize x kernelSize block
				.cwiseProduct(this->kernel)).sum(); //get sum of elementwise product
		}
	}
	return output;
}