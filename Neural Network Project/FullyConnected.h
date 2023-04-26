#pragma once
#include "Layer.h"
#include <Eigen>

//Fully Connected Parameters
struct FC_Parameters {		//need flattening at this point
	int inputChannels = 1;
	int outputChannels = 1;
	//Eigen::MatrixXd weights = Eigen::MatrixXd::Constant(inputChannels, outputChannels, 1);		//want default matrix of 1s to begin with
	Eigen::MatrixXd weights  { { 1, 0, 0, 0},
									   { 0, 1, 0, 0},
									   { 0, 0, 1, 0},
									   { 0, 0, 0, 1} };
	//maybe need weight variable of type vector to perform matrix multiplication 
	//activation function should then be applied to this
	//maybe need bias as well
	//bias should be added to summation before being sent to activation function
	//bias helps to offset the result
};

class FullyConnectedLayer : public BaseLayer {
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
	void setParameters(FC_Parameters params);			

	FC_Parameters getWeight();									
	void setWeights(Eigen::MatrixXd weights);						

												
	Eigen::VectorXd to_linear(Eigen::MatrixXd ip);				//apply activation function then do linearity.
};								
															
									

