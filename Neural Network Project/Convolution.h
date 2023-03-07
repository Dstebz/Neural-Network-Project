#pragma once

#include "ActivationFunction.h"
struct C_Parameters_Default {
	int stride = 1;
	int padding = 1;
	int kernelSize = 3;
	int inputChannels = 1;
	int outputChannels = 1;
};
struct C_Parameters : C_Parameters_Default { //gives default values until overriden
	int stride;
	int padding;
	int kernelSize;
	int inputChannels;
	int outputChannels;
};
class ConvolutionLayer : Layer<C_Parameters> {
private:
	

public:
	//Deconvolution parameters
	
	//Constructors
	ConvolutionLayer();
	ConvolutionLayer(C_Parameters params);
	//Destructor
	~ConvolutionLayer();


	//methods
	void Run(); //Run(Input)
	C_Parameters getParameters(); //should be able to initialise with any Parameter type? change in parent>
	void setParameters(C_Parameters);

	double activationFunction = ActivationFunction::sigmoid();



};

ConvolutionLayer::ConvolutionLayer() { //Empty / default constructor

}

ConvolutionLayer::DeconvolutionLayer(CC_Parameters) {

}