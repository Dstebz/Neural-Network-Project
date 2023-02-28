#pragma once		//prevents the file from being included multiple times

#include "Layer.h"

class PoolingLayer : Layer {
private:
	struct PL_Parameters_Default {
		int stride = 1;				//how much the kernel moves after each cycle
		int padding = 1;			//padding used to assist the kernel in processing of matrix. A padding of 'n' will add an 'n' thick border of 0's
		int kernelSize = 3;			//size of 'filter' being used
		int inputChannels = 1;
		int outputChannels = 1;		//can't we just put this in the default constructor? Will need to do this in the .cpp file

	};

public: 
	//Pooling Parameters
	struct PL_Parameters : PL_Parameters_Default {
		int stride;
		int padding;
		int kernelSize;
		int inputChannels;
		int outputChannels;



	};
	//Pooling constructors
	PoolingLayer();
	PoolingLayer(PL_Parameters params);

	//Pooling destructor
	~PoolingLayer();

	//methods required
	int getMax(/* matrix of kernel */);			//used to calculate the max pooling value

												//also need to use the max values to generate down sized matrix

	void Run(); //takes input and runs
	L_Parameters getParameters(); 
	void setParameters(PL_Parameters);


};

