#pragma once
#include "ActivationFunction.h"
#include "Layer.h"
#include <Eigen>


struct C_Parameters_Default {
	int stride = 1;
	int padding = 1;
	int kernelSize = 3;
	int inputChannels = 1;
	int outputChannels = 1;
};
struct C_Parameters : C_Parameters_Default { //gives default values until overriden
	int stride;
	int padding;
	int kernelSize;
	int inputChannels;
	int outputChannels;
};
class ConvolutionLayer : Layer<C_Parameters> {
private:
	C_Parameters parameters;
	Eigen::MatrixXd kernel;

public:
	//Deconvolution parameters

	//Constructors
	ConvolutionLayer();
	ConvolutionLayer(C_Parameters params);
	//Destructor
	~ConvolutionLayer();


	//Standard Parameters
	Eigen::MatrixXd Run(Eigen::MatrixXd input); //returns dynamic size array of doubles
	C_Parameters getParameters();
	void setParameters(C_Parameters params);

	//Convolution Only
	Eigen::MatrixXd getKernel();
	void setKernel(Eigen::MatrixXd kernel);
	
};

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