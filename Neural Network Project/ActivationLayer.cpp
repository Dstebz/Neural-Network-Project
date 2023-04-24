
#include "ActivationLayer.h"

#include <Eigen>


ActivationLayer::ActivationLayer() = default;

ActivationLayer::ActivationLayer(std::function <Eigen::MatrixXd(Eigen::MatrixXd)> activationFunction) //generic function, takes in a matrix and returns a matrix
{
	this->activationFunction = activationFunction;
}

std::function<Eigen::MatrixXd(Eigen::MatrixXd)> ActivationLayer::getActivationFunction()
{
	return this->activationFunction;
}

void ActivationLayer::setActivationFunction(std::function<Eigen::MatrixXd(Eigen::MatrixXd)> activationFunction)
{
	this->activationFunction = activationFunction;
}
Eigen::MatrixXd ActivationLayer::Run(Eigen::MatrixXd input)
{
	return this->activationFunction(input);
}
