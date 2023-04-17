#pragma once
//include Eigen

#include "Layer.h"
#include "Convolution.h"
#include "Deconvolution.h"
#include "FullyConnected.h"
#include "Pooling.h"

#include <list>
#include <Eigen>
#include <any>

struct NN_Parameters {
	int inputLayerDimensions = 1;
	int hiddenLayerSize = 1;
	int outputLayerDimensions = 1;
};

class NeuralNetwork {
private:
	//Neural Network parameters
	

protected:
	//Neural Network parameters
	NN_Parameters parameters;
	std::list<BaseLayer> hiddenLayers; 


public:
	//constructors
	NeuralNetwork();
	NeuralNetwork(NN_Parameters params);
	NeuralNetwork(std::list<BaseLayer> hiddenLayers); // ANY TO VARIANT
	NeuralNetwork(NN_Parameters params, std::list<BaseLayer> hiddenLayers); // ANY TO VARIANT
	//destructor
	~NeuralNetwork();

	//methods
	Eigen::MatrixXd Run();

	NN_Parameters getParameters();
	void setParameters(NN_Parameters);

	void addLayer(BaseLayer layer, int index);
	void removeLayer(int index);
	std::list<BaseLayer> getLayers();


};
