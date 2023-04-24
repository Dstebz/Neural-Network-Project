//library imports
#include <iostream>
#include <Eigen>

//layer imports
#include "ActivationLayer.h"
#include "Convolution.h"
#include "Deconvolution.h"
#include "FullyConnected.h"
#include "Pooling.h"

//neural network imports 
#include "ActivationFunction.h"
#include "NeuralNetwork.h"



//testing of activation functions
void testActivation() {
    std::cout << "Testing activation functions: " << std::endl;

    Eigen::MatrixXd a{      // construct a 2x2 matrix
      {-3,-0.4 },     // first row
      {0.2,2}      // second row
    };

    std::cout << "Input: " << std::endl;
	std::cout << a << std::endl;

    

    std::cout << "sigmoid function: " << std::endl;

    std::cout << "\n" << "\n";


    Eigen::MatrixXd ans1, ans2, ans3, ans4;

    
    ans1 = reLU(a);
    std::cout << "after applying the reLU function: " << std::endl << ans1 << std::endl;
    std::cout << "\n" << "\n";
    

    ans2 = tanh(a);
    std::cout << "After applying the tanh function: " << std::endl << ans2 << std::endl;
    std::cout << "\n" << "\n";

    ans3 = sigmoid(a);
    std::cout << "After applying the sigmoid function: " << std::endl << ans3 << std::endl;
    std::cout << "\n" << "\n";

    
    ans4 = softmax(a);
    std::cout << "After applying the softmax function: " << std::endl << ans4 << std::endl;
    std::cout << "\n" << "\n";
    


    return;
}


//testing convolution layer
void testConvs()
{
    ConvolutionLayer conv1;
    C_Parameters params{ 1,2,3,4,5 };
    ConvolutionLayer conv2(params);

    std::cout << "conv1 parameters: " << std::endl;
    std::cout << conv1.getParameters().stride << std::endl;
    std::cout << conv1.getParameters().padding << std::endl;
    std::cout << conv1.getParameters().kernelSize << std::endl;

    std::cout << "conv2 parameters:" << std::endl;
    std::cout << conv2.getParameters().stride << std::endl;
    std::cout << conv2.getParameters().padding << std::endl;

    return;

};

//testing deconvolution layer
void testDeconv()
{
    DeconvolutionLayer deconv1;
    return;
};

//testing Neuralnetwork object
void testNN()
{
	NeuralNetwork nn;
	NN_Parameters params{ 2,3,2 };
	nn.setParameters(params);
	nn.addLayer(std::shared_ptr<FullyConnectedLayer>(), 0);
	nn.addLayer(std::shared_ptr<PoolingLayer>(), 1);
    nn.addLayer(std::shared_ptr<ConvolutionLayer>());
	nn.addLayer(std::shared_ptr<DeconvolutionLayer>());
	std::cout << nn.getLayers().size() << std::endl;


    return;
}

int main() { //Neural Networking Testing 
    testNN();
    testActivation();
    testConvs();

    while (true) {}; //loop nothing, keeps window open

    return 0;
}
