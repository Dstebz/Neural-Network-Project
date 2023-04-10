#pragma once
#include "Layer.h"
#include "ActivationFunction.h"
#include <Eigen>


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
	C_Parameters parameters;
	Eigen::MatrixXd kernel;

public:
	//Deconvolution parameters
	
	//Constructors
	ConvolutionLayer();
	ConvolutionLayer(C_Parameters params);
	//Destructor
	~ConvolutionLayer();


	//methods
	Eigen::MatrixXd Run(Eigen::MatrixXd input); //returns dynamic size array of doubles
	C_Parameters getParameters(); 
	void setParameters(C_Parameters params);

};

ConvolutionLayer::ConvolutionLayer() { //Empty / default constructor

}

ConvolutionLayer::ConvolutionLayer(C_Parameters params) {
	
}



//methods
Eigen::MatrixXd ConvolutionLayer::Run(Eigen::MatrixXd input) {
}