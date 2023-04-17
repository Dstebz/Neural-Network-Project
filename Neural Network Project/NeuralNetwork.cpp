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

std::variant<C_Parameters, DC_Parameters, FC_Parameters, PL_Parameters> Parameters;

//CONSTRUCTORS
NeuralNetwork::NeuralNetwork() {
	//Empty / default constructor. Default params, empty Layer List

}

NeuralNetwork::NeuralNetwork(NN_Parameters params) {
	//Constructor with parameters
}

NeuralNetwork::NeuralNetwork(std::list<std::shared_ptr<BaseLayer>> hiddenLayers) {
	//Constructor with hidden layers
}

NeuralNetwork::NeuralNetwork(NN_Parameters params, std::list<std::shared_ptr<BaseLayer>> hiddenLayers) {
	//Constructor with parameters and hidden layers
}

NeuralNetwork::~NeuralNetwork() {
	//Destructor
}

//METHODS

Eigen::MatrixXd NeuralNetwork::Run() {
	//Runs the neural network

	return Eigen::MatrixXd(0, 0);
}

NN_Parameters NeuralNetwork::getParameters() {
	//Returns the parameters of the neural network
	
	return this->parameters;
}

void NeuralNetwork::setParameters(NN_Parameters params) {
	//Sets the parameters of the neural network

	this->parameters = params;
}

void NeuralNetwork::addLayer(BaseLayer layer, int index) {
	//Adds a layer to the neural network
	auto it = this->hiddenLayers.begin(); //.insert requires an iterator
	this->hiddenLayers.insert(std::next(it, index),std::make_shared<BaseLayer>(layer)); //Update to insert
}

void NeuralNetwork::removeLayer(int index) {
	//Removes a layer from the neural network
	auto it = this->hiddenLayers.begin(); //.erase requires an iterator
	this->hiddenLayers.erase(std::next(it,index)); //Update to erase
}

std::list<std::shared_ptr<BaseLayer>> NeuralNetwork::getLayers() {
	//Returns the list of layers in the neural network
	return this->hiddenLayers;
}

