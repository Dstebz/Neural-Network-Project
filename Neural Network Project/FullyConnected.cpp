#include <Eigen>
#include "FullyConnected.h"
#include "ActivationFunction.h"

Eigen::VectorXd FullyConnectedLayer::to_linear(Eigen::MatrixXd ip) {

	Eigen::VectorXd ans = Eigen::Map<Eigen::VectorXd>(ip.data(), ip.size());

	return ans;
}

FC_Parameters FullyConnectedLayer::getParameters() {
	return this->parameters;
}

void FullyConnectedLayer::setParameters(FC_Parameters params) {
	this->parameters = params;
}

FullyConnectedLayer::FullyConnectedLayer() { //Empty / default constructor

}

FullyConnectedLayer::FullyConnectedLayer(FC_Parameters params) {

}


