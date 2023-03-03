#pragma once

#include <string>
#include <variant>

#include "Deconvolution.h"
#include "Convolution.h"
#include "Pooling.h"
#include "FullyConnected.h"


class Layer { //interface class

protected:
	std::variant<DeconvolutionLayer::DC_Parameters, ConvolutionLayer::C_Parameters, PoolingLayer::PL_Parameters, FullyConnectedLayer::FC_Parameters> L_Parameters;
public:
	//constructors
	Layer();
	//destructor
	~Layer();
	
	virtual void Run(); //return matrix / image?
	virtual getParameters();
	virtual void setParameters(L_Parameters); 

};

Layer::Layer() {
	//Empty / default constructor

}

Layer::~Layer() {
	//Destructor
}
