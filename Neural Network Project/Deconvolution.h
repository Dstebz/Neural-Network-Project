#pragma once

#include "Layer.h"

#include "Convolution.h"
#include "Deconvolution.h"
#include "FullyConnected.h"
#include "Pooling.h"


#include <Eigen>


struct DC_Parameters {
	int stride = 1;
	int padding = 1;
	int kernelSize = 3;
	int inputChannels = 1;
	int outputChannels = 1;
};
class DeconvolutionLayer : public BaseLayer {
private:

	DC_Parameters parameters;
	Eigen::MatrixXd kernel;

public:
	//Constructors
	DeconvolutionLayer();
	DeconvolutionLayer(DC_Parameters params);
	DeconvolutionLayer(DC_Parameters params, Eigen::MatrixXd kernel);

	//Destructor
	//~DeconvolutionLayer();

	//Parameters
	//methods
	Eigen::MatrixXd Run(); //Run(Input)
	DC_Parameters getParameters(); //should be able to initialise with any Parameter type? change in parent>
	void setParameters(DC_Parameters); 

	void setKernel(Eigen::MatrixXd kernel);
	Eigen::MatrixXd getKernel();
	Eigen::MatrixXd Run(Eigen::MatrixXd input);
};
