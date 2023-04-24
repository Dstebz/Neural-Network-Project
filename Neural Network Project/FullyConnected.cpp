
#include "FullyConnected.h"
#include <Eigen>


FC_Parameters FullyConnectedLayer::getParameters() {
	return this->parameters;
};

void FullyConnectedLayer::setParameters(FC_Parameters params) {
	this->parameters = params;
	return;
};

FC_Parameters FullyConnectedLayer::getWeight() {			//not sure 
	//unsure of output or what to put in code;
	return this->parameters;
};

void FullyConnectedLayer::setWeights(Eigen::MatrixXd weights) {
	if (weights.rows() != this->parameters.inputChannels || weights.cols() != this->parameters.outputChannels) {
		throw std::invalid_argument("Invalid weight matrix dimensions");
	}
	else
	{
		this->parameters.weights = weights;
	}
	
	return;

};

FullyConnectedLayer::FullyConnectedLayer() = default;

FullyConnectedLayer::FullyConnectedLayer(FC_Parameters const params) { //constructor with parameters
	//not sure what to include in here
	this->parameters = params;
};

FullyConnectedLayer::~FullyConnectedLayer() = default; 

Eigen::MatrixXd FullyConnectedLayer::Run(Eigen::MatrixXd input) {
	
	Eigen::MatrixXd ans = input * this->parameters.weights;

	return ans;
};