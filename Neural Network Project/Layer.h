
#pragma once

#include <string>
#include <variant>
#include <Eigen>

template <class T>

class Layer { //interface class

protected:
public:
	
	virtual Eigen::MatrixXd Run(Eigen::MatrixXd input); //return matrix / image?
	virtual T getParameters() = 0;
	virtual void setParameters(T params) = 0; 

	virtual double activationFunction();

};
