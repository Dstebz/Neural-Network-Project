#pragma once		//prevents the file from being included multiple times

#include "Layer.h"

struct PL_Parameters_Default {
	int filter = 1;
	int output_channels = 1; 
	int input_channels = 1;
	int stride = 1;
};

struct PL_Parameters : PL_Parameters_Default {
	int filter;
	int output_channels;
	int input_channels;
	int stride;

};

class PoolingLayer : Layer<PL_Parameters> {		//pooling layer used to reduce dimension of feature map, reducing number of parameters to learn...
private:							//max pooling used to capture most dominant aspects of feature map (i.e. will choose largest value in quadrant of matrix)
	

public: 
	//Pooling Parameters
	
	//Pooling constructors
	PoolingLayer();
	PoolingLayer(PL_Parameters params);

	//Pooling destructor
	~PoolingLayer();

	//methods required
	Eigen::MatrixXd max_pool(/* input matrix */ int filter, int stride);	

	Eigen::MatrixXd avg_pool(/* input matrix */ int filter, int stride);

	Eigen::MatrixXd global_pool(/* input matrix */);

	//maybe include method that checks whether pooling is possible due to dimensions of filter, stride, input matrix		
	
	void filter_limits(/* input matrix */ int filter, int stride)		//checks whether pooling method is possible 
																		//i.e. stride cant be more than input_width - filter_width
																		//check logbook for more info

	void Run(); //takes input and runs
	PL_Parameters getParameters(); 
	void setParameters(PL_Parameters);


};

