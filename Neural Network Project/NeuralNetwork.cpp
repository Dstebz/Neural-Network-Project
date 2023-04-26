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
	auto  cascadeRun = [&runningOutput](std::shared_ptr<BaseLayer> const& layer) { //grab runningoutput from enclosing scope
		runningOutput = layer->Run(runningOutput); // run layer, save output in runningOutput.
	};
	for_each(this->hiddenLayers.begin(), this->hiddenLayers.end(), cascadeRun); //for each layer, run the lambda function

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

