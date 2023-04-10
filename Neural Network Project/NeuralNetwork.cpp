//include Eigen


#include "NeuralNetwork.h"

#include <vector>
#include <Eigen>

#include <any>
#include <variant>

std::variant<C_Parameters, DC_Parameters, FC_Parameters, PL_Parameters> Parameters;

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