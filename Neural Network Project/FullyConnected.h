#pragma once

#include "Layer.h"
struct FC_Parameters_Default {

	//	int stride = 1;				//how much the kernel moves after each cycle
	//	int padding = 1;			//padding used to assist the kernel in processing of matrix. A padding of 'n' will add an 'n' thick border of 0's
	//	int kernelSize = 3;			//size of 'filter' being used
	//	int inputChannels = 1;
	//	int outputChannels = 1;		//can't we just put this in the default constructor? Will need to do this in the .cpp file

	int inputChannels = 1;
	int batchSize = 1;
	int outputChannels = 1;


};
//Fully Connected Parameters
struct FC_Parameters : FC_Parameters_Default { //gives default values until overriden
	/*
	int stride;
	int padding;
	int kernelSize;
	int inputChannels;
	int outputChannels;
	*/

	int inputChannels;
	int outputChannels;
	int batchSize;				/* choosing appropriate batch size means that memory usage is balanced well with training speed
								- batch size of around 32 should be sufficient, can be decrease it for accuracy or increase it for efficiency
								*/

								//maybe need weight variable of type vector to perfrom matrix multiplication 
								//activation function should then be applied to this
								//maybe need bias as well
								//bias should be added to summation before being sent to activation function
								//bias helps to offset the result
};

class FullyConnectedLayer : Layer<FC_Parameters> {
private: 
	

public: 
	

	//constructors

	FullyConnectedLayer();
	FullyConnectedLayer(FC_Parameters params);

	//Destructor

	~FullyConnectedLayer();


	//Also need to add the final output layer

};