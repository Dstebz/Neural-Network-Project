#pragma once
//include Eigen

#include "NeuralNetwork.h"
#include "Layer.h"

#include <vector>
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
	std::vector<std::unique_ptr<Layer<std::any>>> hiddenLayers; //unique_ptr is a smart pointer that deletes itself when it goes out of scope ANY TO VARIANT


public:
	//constructors
	NeuralNetwork();
	NeuralNetwork(NN_Parameters params);
	NeuralNetwork(std::vector<std::unique_ptr<Layer<std::any>>> hiddenLayers); // ANY TO VARIANT
	NeuralNetwork(NN_Parameters params, std::vector<std::unique_ptr<Layer<std::any>>> hiddenLayers); // ANY TO VARIANT
	//destructor
	~NeuralNetwork();

	//methods
	void Run();

	NN_Parameters getParameters();
	void setParameters(NN_Parameters);

	void addLayer(Layer<std::any> layer);
	void removeLayer(int index);
	std::vector<Layer<std::any>> getLayers();


};
