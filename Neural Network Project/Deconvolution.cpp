
#include "Deconvolution.h"
#include "ActivationFunction.h"

#include <Eigen>

DeconvolutionLayer::DeconvolutionLayer() {}; //Empty / default constructor

DeconvolutionLayer::DeconvolutionLayer(DC_Parameters params) {
	this->parameters = params;
	this->kernel = Eigen::MatrixXd::Constant(this->parameters.kernelSize, this->parameters.kernelSize,1);
};

DeconvolutionLayer::DeconvolutionLayer(DC_Parameters params, Eigen::MatrixXd kernel) {
	this->parameters = params;
	this->kernel = kernel;
};

//methods
DeconvolutionLayer::DC_Parameters DeconvolutionLayer::getParameters() {
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


Eigen::MatrixXd DeconvolutionLayer::Run(Eigen::MatrixXd input) {
	int outputX = (input.rows() - 1) * this->parameters.stride + this->parameters.kernelSize - 2 * this->parameters.padding;
	int outputY = (input.cols() - 1) * this->parameters.stride + this->parameters.kernelSize - 2 * this->parameters.padding;
	Eigen::MatrixXd output = Eigen::MatrixXd::Zero(outputX, outputY);

	for (int i = 0; i < input.rows(); i++) {
		for (int j = 0; j < input.cols(); j++) {
			output.block(i * this->parameters.stride, j * this->parameters.stride, this->parameters.kernelSize, this->parameters.kernelSize) += input(i, j) * this->kernel;
		}
	}
	return output;
};

