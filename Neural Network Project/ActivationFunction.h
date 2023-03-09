#pragma once
#include <Eigen>


class ActivationFunction {
public:
	
	Eigen::MatrixXd reLU(Eigen::MatrixXd ip);			//maybe include this within layers since it's not taking up much space
	Eigen::MatrixXd tanh(Eigen::MatrixXd ip);	
	Eigen::MatrixXd sigmoid(Eigen::MatrixXd ip);
	Eigen::MatrixXd softmax(Eigen::MatrixXd ip);

};