#pragma once		//prevents the file from being included multiple times

#include "Layer.h"
#include <Eigen>
#include <string>

struct PL_Parameters {
	int filter = 1;
	int output_channels = 1;
	int input_channels = 1;
	int stride = 1;
	std::string pooling_type = "max";

};

class PoolingLayer : public BaseLayer {		//pooling layer used to reduce dimension of feature map, reducing number of parameters to learn...
private:							//max pooling used to capture most dominant aspects of feature map (i.e. will choose largest value in quadrant of matrix)
	
	
	//Pooling Parameters
	
	PL_Parameters parameters;

public:
	
	//Pooling Parameters
	
	//Pooling constructors
	PoolingLayer();
	PoolingLayer(PL_Parameters params);

	//Pooling destructor
	~PoolingLayer();

	//methods required
	Eigen::MatrixXd max_pool(Eigen::MatrixXd input, int filter, int stride);

	Eigen::MatrixXd avg_pool(Eigen::MatrixXd input, int filter, int stride);

	double global_pool(Eigen::MatrixXd input, std::string global_pool_type);

	//maybe include method that checks whether pooling is possible due to dimensions of filter, stride, input matrix		
	
	Eigen::MatrixXd Run(Eigen::MatrixXd input); //takes input and runs
	PL_Parameters getParameters();                        //should be defined in Layer.h
	void setParameters(PL_Parameters);

};

