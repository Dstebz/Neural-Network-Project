#pragma once

#include <map>
#include <any> //generics for changing parameters
#include "Layer.h"

struct DC_Parameters_Default {
	int stride = 1;
	int padding = 1;
	int kernelSize = 3;
	int inputChannels = 1;
	int outputChannels = 1;
};
struct DC_Parameters : DC_Parameters_Default { //gives default values until overriden
	int stride;
	int padding;
	int kernelSize;
	int inputChannels;
	int outputChannels;
};

class DeconvolutionLayer : Layer<DC_Parameters>{
private:
	

public:
	//Deconvolution parameters
	
	//Constructors
	DeconvolutionLayer();
	DeconvolutionLayer(DC_Parameters params);
	//Destructor
	~DeconvolutionLayer();


	//methods
	void Run(); //Run(Input)
	DC_Parameters getParameters(); //should be able to initialise with any Parameter type? change in parent>
	void setParameters(DC_Parameters); 
	


};

DeconvolutionLayer::DeconvolutionLayer() { //Empty / default constructor

}

DeconvolutionLayer::DeconvolutionLayer(DC_Parameters) {
	
}