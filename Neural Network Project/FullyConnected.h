#pragma once

#include "Layer.h"

class FullyConnectedLayer : Layer {
private: 
	struct FC_Parameters_Default {

		int stride = 1;				//how much the kernel moves after each cycle
		int padding = 1;			//padding used to assist the kernel in processing of matrix. A padding of 'n' will add an 'n' thick border of 0's
		int kernelSize = 3;			//size of 'filter' being used
		int inputChannels = 1;
		int outputChannels = 1;		//can't we just put this in the default constructor? Will need to do this in the .cpp file
		
	};

public: 
	//Fully Connected Parameters
	struct FC_Parameters : FC_Parameters_Default { //gives default values until overriden
		int stride;
		int padding;
		int kernelSize;
		int inputChannels;
		int outputChannels;
	};

	//constructors

	FullyConnectedLayer();
	FullyConnectedLayer(FC_Parameters params);

	//Destructor

	~FullyConnectedLayer();


	//Also need to add the final output layer

};