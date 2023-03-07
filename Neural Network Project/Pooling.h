#pragma once		//prevents the file from being included multiple times

#include "Layer.h"

class PoolingLayer : Layer {		//pooling layer used to reduce dimension of feature map, reducing number of parameters to learn...
private:							//max pooling used to capture most dominant aspects of feature map (i.e. will choose largest value in quadrant of matrix)
	struct PL_Parameters_Default {
		int map_width;
		int map_height;
		int n_channels;
		int filter_size;
		int stride;
	};

public: 
	//Pooling Parameters
	struct PL_Parameters : PL_Parameters_Default {
		int map_width;
		int map_height;
		int n_channels;
		int filter_size;
		int stride;

	};
	//Pooling constructors
	PoolingLayer();
	PoolingLayer(PL_Parameters params);

	//Pooling destructor
	~PoolingLayer();

	//methods required
	int getMax(/* matrix of kernel, */ int map_width, int map_height, int filter_size, int stride);			//used to calculate the max pooling value

												//also need to use the max values to generate down sized matrix

	void Run(); //takes input and runs
	L_Parameters getParameters(); 
	void setParameters(PL_Parameters);


};

