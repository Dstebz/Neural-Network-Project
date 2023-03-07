#pragma once

#include <string>
#include <variant>

#include "Deconvolution.h"
#include "Convolution.h"
#include "Pooling.h"
#include "FullyConnected.h"
#include <Eigen>

template <typename ParameterStruct>
class Layer { //interface class

protected:
public:
	//constructors
	Layer();
	//destructor
	~Layer();
	
	virtual Eigen::Matrix Run(Eigen::Matrix input); //return matrix / image?
	virtual ParameterStruct getParameters() = 0;
	virtual void setParameters(ParameterStruct params) = 0; 

	virtual double activationFunction();

};
