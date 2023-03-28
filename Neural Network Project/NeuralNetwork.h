#pragma once
//include Eigen

#include <vector>
#include "Layer.h"
#include <Eigen>




class NeuralNetwork {
private:
	//Neural Network parameters
	struct NN_Parameters {
		int inputLayerDimensions = 1;
		int hiddenLayerSize = 1;
		int outputLayerDimensions = 1;
	};

protected:
	//Neural Network parameters
	NN_Parameters parameters;
	std::vector<Layer<T>*> hiddenLayers;

	
public:
	//constructors
	NeuralNetwork();
	NeuralNetwork(NN_Parameters params);
	NeuralNetwork(std::vector<Layer> hiddenLayers);
	NeuralNetwork(NN_Parameters params, std::vector<Layer> hiddenLayers);
	//destructor
	~NeuralNetwork();

	//methods
	void Run();
	
	NN_Parameters getParameters();
	void setParameters(NN_Parameters);
	
	void addLayer(Layer layer);
	void removeLayer(int index);
	std::vector<Layer<std::any>> getLayers();
	
	
};

//CONSTRUCTORS
NeuralNetwork::NeuralNetwork() {
	//Empty / default constructor. Default params, empty Layer List
	
}

NeuralNetwork::NeuralNetwork(NN_Parameters params) {
	//Constructor with parameters
}

NeuralNetwork::NeuralNetwork(std::vector<Layer> hiddenLayers) {
	//Constructor with hidden layers
}

NeuralNetwork::NeuralNetwork(NN_Parameters params, std::vector<Layer> hiddenLayers) {
	//Constructor with parameters and hidden layers
}

NeuralNetwork::~NeuralNetwork() {
	//Destructor
}