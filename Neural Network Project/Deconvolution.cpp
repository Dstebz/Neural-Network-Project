
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
	int outputX = (input.rows() - 1) * this->parameters.stride + this->parameters.kernelSize - 2 * this->parameters.padding;
	int outputY = outputX;

	int strideDimensions = input.rows()+(input.rows() - 1) * this->parameters.stride; //dimensions of matrix after stride
	int paddedDimensions = strideDimensions + 2 * this->parameters.padding; //dimensions of matrix after padding
	Eigen::MatrixXd paddedinput = Eigen::MatrixXd::Zero(paddedDimensions, paddedDimensions);

	for (int i = 0; i < input.rows(); i++) {
		for (int j = 0; j < input.cols(); j++) {
			paddedinput(i*(this->parameters.stride+1) + this->parameters.padding, j * (this->parameters.stride + 1) + this->parameters.padding) = input(i, j);
		}
	}
	Eigen::MatrixXd output = Eigen::MatrixXd::Zero(outputX, outputY);


	//Regular convolution code below
	int scanLength = paddedinput.rows() - this->parameters.kernelSize + 1; //length of scan

	//debug prints
	std::cout << "Padding size: " << this->parameters.padding << "\n";
	std::cout << "scanLength: " << scanLength << "\n";
	std::cout << "OutputX: " << outputX << "\n";
	std::cout << "padded dimensions: " << paddedDimensions << "\n";
	std::cout << "padded input: " << std::endl << paddedinput << "\n";

	for (int i = 0; i < scanLength; i++) {
		for (int j = 0; j < scanLength; j++) {
			std::cout << "i: " << i << " j: " << j << "\n";
			
			output(i, j) = (paddedinput.block(i, //stride always 1 for deconvolution
				j, //stride always 1 for deconvolution
				this->parameters.kernelSize,
				this->parameters.kernelSize) //take kernelSize x kernelSize block
				.cwiseProduct(this->kernel)).sum(); //get sum of elementwise product
		}
	}
	return output;


};

