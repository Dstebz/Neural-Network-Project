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
const C_Parameters ConvolutionLayer::getParameters() {
	return this->parameters;
};
void ConvolutionLayer::setParameters(C_Parameters params) {
	this->parameters = params;
};
Eigen::MatrixXd ConvolutionLayer::getKernel() {
	return this->kernel;
};
void ConvolutionLayer::setKernel(Eigen::MatrixXd kernel) {
	this->kernel = kernel;
}

Eigen::MatrixXd ConvolutionLayer::Run(Eigen::MatrixXd input) {
	int outputDimensionX = (input.rows() -  2 * this->parameters.padding) / this->parameters.stride;
	int outputDimensionY = (input.cols() - 2 * this->parameters.padding) / this->parameters.stride;
	
	Eigen::MatrixXd output = Eigen::MatrixXd::Zero(outputDimensionX, outputDimensionY);

	//adding padding to input
	Eigen::MatrixXd paddedinput = Eigen::MatrixXd::Zero(input.rows() + 2 * this->parameters.padding, //add padding to rows
		input.cols() + 2 * this->parameters.padding); //add padding to columns
	paddedinput.block(this->parameters.padding, //start at padding, padding
		this->parameters.padding,
		input.rows(), //take input row dimension
		input.cols() //take input column dimension
	) = input; //assign input into the padded matrix


	for (int i = 0; i < outputDimensionX; i++) {
		for (int j = 0; j < outputDimensionY; j++) {
			output(i, j) = (paddedinput.block(i * this->parameters.stride, //start at i*stride, j*stride (top left corner of kernel)
				j * this->parameters.stride,
				this->parameters.kernelSize,
				this->parameters.kernelSize) //take kernelSize x kernelSize block
				.cwiseProduct(this->kernel)).sum(); //get sum of elementwise product
		}
	}
	return output;
}