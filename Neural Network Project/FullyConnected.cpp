
#include "FullyConnected.h"
#include <Eigen>
/*
Eigen::VectorXd FullyConnectedLayer::to_linear(Eigen::MatrixXd ip) {

	Eigen::VectorXd ans = Eigen::Map<Eigen::VectorXd>(ip.data(), ip.size());
	return ans;
}
probs don't need rn
*/

FC_Parameters FullyConnectedLayer::getParameters() {
	return this->parameters;
};

void FullyConnectedLayer::setParameters(FC_Parameters params) {
	this->parameters = params;
};

FC_Parameters FullyConnectedLayer::getWeight() {			//not sure 
	//unsure of output or what to put in code;
	return this->parameters;
};

void FullyConnectedLayer::setWeight(FC_Parameters const weights) {

};

FullyConnectedLayer::FullyConnectedLayer() = default;

FullyConnectedLayer::FullyConnectedLayer(FC_Parameters const params) { //constructor with parameters
	//not sure what to include in here
	this->parameters = params;
};

FullyConnectedLayer::~FullyConnectedLayer() = default; 

Eigen::MatrixXd FullyConnectedLayer::Run(Eigen::MatrixXd input) {
	
	Eigen::MatrixXd ans = input * this->parameters.weight;

	return ans;
};