#pragma once

#include "Deconvolution.h"
#include "ActivationFunction.h"

#include <Eigen>

struct DC_Parameters_Default {
	int stride = 1;
	int padding = 1;
	int kernelSize = 3;
	int inputChannels = 1;
	int outputChannels = 1;
};
struct DC_Parameters : DC_Parameters_Default { //gives default values until overriden
	int stride;
	int padding;
	int kernelSize;
	int inputChannels;
	int outputChannels;
};

DeconvolutionLayer::DeconvolutionLayer() {} //Empty / default constructor

DeconvolutionLayer::DeconvolutionLayer(DC_Parameters params) {
	this->parameters = params;
	this->kernel = Eigen::MatrixXd::Random(this->parameters.kernelSize, this->parameters.kernelSize);
}
DeconvolutionLayer::DeconvolutionLayer(DC_Parameters params, Eigen::MatrixXd kernel) {}

//methods
DC_Parameters DeconvolutionLayer::getParameters() {}
void DeconvolutionLayer::setParameters(DC_Parameters params) {}

Eigen::MatrixXd DeconvolutionLayer::getKernel() {}

void DeconvolutionLayer::setKernel(Eigen::MatrixXd kernel) {}


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
}

