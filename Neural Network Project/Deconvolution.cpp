#pragma once

#include "Layer.h"
#include "Deconvolution.h"

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

class DeconvolutionLayer : Layer<DC_Parameters> {
private:
	DC_Parameters parameters;
	Eigen::MatrixXd kernel;

public:
	//Deconvolution parameters

	//Constructors
	DeconvolutionLayer();
	DeconvolutionLayer(DC_Parameters params);
	DeconvolutionLayer(DC_Parameters params, Eigen::MatrixXd kernel);
	//Destructor
	~DeconvolutionLayer();


	//methods
	Eigen::MatrixXd Run(Eigen::MatrixXd input); //returns dynamic size array of doubles
	DC_Parameters getParameters(); //should be able to initialise with any Parameter type? change in parent>
	void setParameters(DC_Parameters);

	Eigen::MatrixXd getKernel();
	void setKernel(Eigen::MatrixXd kernel);
	
	
};

DeconvolutionLayer::DeconvolutionLayer() { //Empty / default constructor
	this->kernel = Eigen::MatrixXd::Constant(this->parameters.kernelSize, this->parameters.kernelSize, 1); //square Matrix, initialised with 1s

}

DeconvolutionLayer::DeconvolutionLayer(DC_Parameters params) {
	this->parameters = params;
	this->kernel = Eigen::MatrixXd::Constant(this->parameters.kernelSize, this->parameters.kernelSize, 1); //square Matrix, initialised with 1s
}

DeconvolutionLayer::DeconvolutionLayer(DC_Parameters params, Eigen::MatrixXd kernel) {
	this->parameters = params;
	this->kernel = kernel;
}

//methods
DC_Parameters DeconvolutionLayer::getParameters() {
	return this->parameters;
}
void DeconvolutionLayer::setParameters(DC_Parameters params) {
	this->parameters = params;
}

Eigen::MatrixXd DeconvolutionLayer::getKernel() {
	return this->kernel;
}

void DeconvolutionLayer::setKernel(Eigen::MatrixXd kernel) {
	this->kernel = kernel;
}


Eigen::MatrixXd DeconvolutionLayer::Run(Eigen::MatrixXd input) {
}

