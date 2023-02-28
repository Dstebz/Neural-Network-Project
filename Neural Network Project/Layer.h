#pragma once

#include <string>

#include "Deconvolution.h"
#include "Convolution.h"


class Layer { //interface class

protected:
	union L_Parameters {
		DeconvolutionLayer::DC_Parameters deconvolution;
		//ConvolutionLayer::C_Parameters convolution;
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
