#pragma once

#include <string>
#include <variant>

#include "Deconvolution.h"
#include "Convolution.h"
#include "Pooling.h"
#include "FullyConnected.h"


class Layer { //interface class

protected:
	union L_Parameters{
		DeconvolutionLayer::DC_Parameters;
		ConvolutionLayer::C_Parameters;
		PoolingLayer::PL_Parameters;
		FullyConnectedLayer::FC_Parameters;
	};
public:
	//constructors
	Layer();
	Layer(L_Parameters);
	//destructor
	~Layer();
	
	virtual void Run(); //return matrix / image?
	virtual L_Parameters getParameters();
	virtual void setParameters(L_Parameters); 

};

Layer::Layer() {
	//Empty / default constructor

}

Layer::~Layer() {
	//Destructor
}
