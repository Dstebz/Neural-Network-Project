//include Eigen

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Convolution.h"
#include "Deconvolution.h"
#include "FullyConnected.h"
#include "Pooling.h"

#include <list>
#include <Eigen>
#include <variant>
#include <iterator>
#include <iostream>

std::variant<C_Parameters, DC_Parameters, FC_Parameters, PL_Parameters> Parameters;

//CONSTRUCTORS
NeuralNetwork::NeuralNetwork() {
	//Empty / default constructor. Default params, empty Layer List

}

NeuralNetwork::NeuralNetwork(NN_Parameters params) {
	//Constructor with parameters
	this->parameters = params;
};

NeuralNetwork::NeuralNetwork(std::list<std::shared_ptr<BaseLayer>> hiddenLayers) {
	//Constructor with hidden layers
	this->hiddenLayers = hiddenLayers;
};

NeuralNetwork::NeuralNetwork(NN_Parameters params, std::list<std::shared_ptr<BaseLayer>> hiddenLayers) {
	//Constructor with parameters and hidden layers
	this->parameters = params;
	this->hiddenLayers = hiddenLayers;
};

NeuralNetwork::~NeuralNetwork() {
	//Destructor
};

//METHODS

Eigen::MatrixXd NeuralNetwork::Run(Eigen::MatrixXd const input) {
	//Runs the neural network
	Eigen::MatrixXd runningOutput = input;
	for (std::shared_ptr<BaseLayer> const layer : this->hiddenLayers) { //layer not edited, so const
		try
		{
			runningOutput = layer->Run(runningOutput); //pass the result of each run to the next layer
		}
		catch (const std::exception&)
		{
			std::cout << "Error: Layer run failed" << std::endl;
		}

	};
	return runningOutput;

};

NN_Parameters NeuralNetwork::getParameters() {
	//Returns the parameters of the neural network

	return this->parameters;
};

void NeuralNetwork::setParameters(NN_Parameters const params) {
	//Sets the parameters of the neural network

	this->parameters = params;
};

void NeuralNetwork::addLayer(std::shared_ptr<BaseLayer> const layer) {
	//Adds a layer to the neural network
	this->hiddenLayers.push_back(layer);
};
void NeuralNetwork::addLayer(std::shared_ptr<BaseLayer> const layer, int const index) {
	//Adds a layer to the neural network
	auto it = this->hiddenLayers.begin(); //.insert requires an iterator
	this->hiddenLayers.insert(std::next(it, index), layer); //Update to insert
};

void NeuralNetwork::removeLayer(int const index) {
	//Removes a layer from the neural network
	auto it = this->hiddenLayers.begin(); //.erase requires an iterator
	this->hiddenLayers.erase(std::next(it, index)); //Update to erase
};

std::list<std::shared_ptr<BaseLayer>> NeuralNetwork::getLayers() {
	//Returns the list of layers in the neural network
	return this->hiddenLayers;
};

