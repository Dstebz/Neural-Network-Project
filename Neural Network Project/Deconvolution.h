#pragma once

#include <map>
#include <any> //generics for changing parameters
#include "Layer.h"


class DeconvolutionLayer : Layer{
private:
	struct Parameters {
		int stride = 1;
		int padding = 1;
		int kernelSize = 3;
		int inputChannels = 1;
		int outputChannels = 1;
	};

public:
	DeconvolutionLayer();
	DeconvolutionLayer(Parameters params);
	~DeconvolutionLayer();

	void Run();
	std::any getParameters(); //should be able to initialise with any Parameter type? change in parent>
	void setParameters(Parameters); 


};

DeconvolutionLayer::DeconvolutionLayer() { //Empty / default constructor
	DeconvolutionLayer(Parameters);
}

DeconvolutionLayer::DeconvolutionLayer(Parameters) {
	
}