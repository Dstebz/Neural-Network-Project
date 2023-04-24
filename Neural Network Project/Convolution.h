#pragma once

#include "Layer.h"
#include "ActivationFunction.h"
#include <Eigen>

struct C_Parameters {
	int stride = 1;
	int padding = 1;
	int kernelSize = 3;
	int inputChannels = 1;
	int outputChannels = 1;
};


class ConvolutionLayer : public BaseLayer {
private:
	
	
	C_Parameters parameters;
	Eigen::MatrixXd kernel;

public:
	//Convolution parameters
	
	//Constructors
	ConvolutionLayer();
	ConvolutionLayer(C_Parameters params);

	//Destructor
	~ConvolutionLayer() override;

	//methods
	Eigen::MatrixXd Run(Eigen::MatrixXd input); //returns dynamic size array of doubles
	const C_Parameters getParameters(); 
	void setParameters(C_Parameters params);
	void setKernel(Eigen::MatrixXd kernel);
	Eigen::MatrixXd getKernel();
};

//methods