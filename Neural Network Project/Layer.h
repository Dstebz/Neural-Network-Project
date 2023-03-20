
#pragma once

#include <string>
#include <variant>
#include <Eigen>

#include "Deconvolution.h"
#include "Convolution.h"
#include "Pooling.h"
#include "FullyConnected.h"


template <typename T>

class Layer { //interface class

protected:
public:
	//constructors
	Layer();
	//destructor
	~Layer();
	
	virtual Eigen::MatrixXd Run(Eigen::MatrixXd input); //return matrix / image?
	virtual T getParameters() = 0;
	virtual void setParameters(T params) = 0; 

	virtual double activationFunction();

};
