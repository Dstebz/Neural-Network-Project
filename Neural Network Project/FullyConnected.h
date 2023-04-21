#pragma once
#include "Layer.h"
#include <Eigen>

struct FC_Parameters {		//need flattening at this point
	int inputChannels = 1;
	int outputChannels = 1;
	Eigen::MatrixXd weight = Eigen::MatrixXd::Constant(inputChannels, outputChannels, 1);		//want default matrix of 1s to begin with
	/* choosing appropriate batch size means that memory usage is balanced well with training speed
	- batch size of around 32 should be sufficient, can be decrease it for accuracy or increase it for efficiency
	*/

	//maybe need weight variable of type vector to perfrom matrix multiplication 
	//activation function should then be applied to this
	//maybe need bias as well
	//bias should be added to summation before being sent to activation function
	//bias helps to offset the result
};


//Fully Connected Parameters

class FullyConnectedLayer : BaseLayer {
private:
	
	FC_Parameters parameters;

public:
	


	//constructors
	FullyConnectedLayer();
	FullyConnectedLayer(FC_Parameters params);
	//FullyConnectedLayer(FC_Parameters params, Eigen::MatrixXd weights);	maybe no need for this since already in FC_Parameters struct

	//Destructor
	~FullyConnectedLayer();

	//methods
	Eigen::MatrixXd Run(Eigen::MatrixXd input); 
	FC_Parameters getParameters();								//setters and getters
	void setParameters(FC_Parameters);			

	FC_Parameters getWeight();									//unsure of inputs...
	void setWeight(FC_Parameters weights);						//unsure of inputs...

												
	Eigen::VectorXd to_linear(Eigen::MatrixXd ip);				//apply activation function then do linearity.
};								
															
									

