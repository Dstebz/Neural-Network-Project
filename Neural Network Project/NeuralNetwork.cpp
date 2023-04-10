#pragma once
//include Eigen

#include <vector>
#include "NeuralNetwork.h"
#include "Layer.h"
#include <Eigen>

#include <any>

//CONSTRUCTORS
NeuralNetwork::NeuralNetwork() {
	//Empty / default constructor. Default params, empty Layer List

}

NeuralNetwork::NeuralNetwork(NN_Parameters params) {
	//Constructor with parameters
}

NeuralNetwork::NeuralNetwork(std::vector<std::unique_ptr<Layer<std::any>>> hiddenLayers) {
	//Constructor with hidden layers
}

NeuralNetwork::NeuralNetwork(NN_Parameters params, std::vector<std::unique_ptr<Layer<std::any>>> hiddenLayers) {
	//Constructor with parameters and hidden layers
}

NeuralNetwork::~NeuralNetwork() {
	//Destructor
}