#pragma once
class ConvolutionLayer : Layer {
private:
	struct C_Parameters_Default {
		int stride = 1;
		int padding = 1;
		int kernelSize = 3;
		int inputChannels = 1;
		int outputChannels = 1;
	};

public:
	//Deconvolution parameters
	struct C_Parameters : C_Parameters_Default { //gives default values until overriden
		int stride;
		int padding;
		int kernelSize;
		int inputChannels;
		int outputChannels;
	};
	//Constructors
	ConvolutionLayer();
	ConvolutionLayer(C_Parameters params);
	//Destructor
	~ConvolutionLayer();


	//methods
	void Run(); //Run(Input)
	L_Parameters getParameters(); //should be able to initialise with any Parameter type? change in parent>
	void setParameters(C_Parameters);



};

ConvolutionLayer::ConvolutionLayer() { //Empty / default constructor

}

ConvolutionLayer::DeconvolutionLayer(CC_Parameters) {

}