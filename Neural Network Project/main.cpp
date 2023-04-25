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
	//testing parameters
    std::cout << "Testing convolution layers: " << std::endl;
    ConvolutionLayer conv1;
    C_Parameters params{ 1,2,3,4,5 };
    ConvolutionLayer conv2(params);
	
    std::cout << "conv1 parameters: " << std::endl;
    std::cout << "Stride" << conv1.getParameters().stride << std::endl;
    std::cout << "Padding" << conv1.getParameters().padding << std::endl;
    std::cout << "Kernel size" << conv1.getParameters().kernelSize << std::endl;
	std::cout << "Expect: 1,1,3" << std::endl;

    std::cout << "conv2 parameters:" << std::endl;
    std::cout << "Stride" << conv2.getParameters().stride << std::endl;
    std::cout << "Padding" << conv2.getParameters().padding << std::endl;
	std::cout << "Kernel size" << conv2.getParameters().kernelSize << std::endl;
	std::cout << "Expect: 1,2,3" << std::endl;

	std::cout << "Testing Convolution Kernel: " << std::endl;
	ConvolutionLayer conv3;
	std::cout << "Default Kernel: " << std::endl;
	std::cout << conv3.getKernel() << std::endl;
	std::cout << "Expect: 3x3 matrix of 1s" << std::endl;

    Eigen::MatrixXd testKernel = Eigen::MatrixXd::Zero(3, 3);
	testKernel <<
				0, 0, 0,
				0, 1, 0,
				0, 0, 0; //Identity kernel, does not change input

    conv3.setKernel(testKernel);

    std::cout << "Test kernel: " << std::endl;
    std::cout << conv3.getKernel() << std::endl;
    std::cout << "Expect 3x3, 0s on edge, 1 in centre" << std::endl;

    std::cout << "Testing Convolution Run():" << std::endl;

    Eigen::MatrixXd testInput = Eigen::MatrixXd::Constant(4, 4, 2);
    conv3.setParameters({ 1,1,3,0,0 });
    Eigen::MatrixXd convolved = conv3.Run(testInput);

    std::cout << convolved << std::endl;
    std::cout << "Expect 4x4 of 2s" << std::endl;





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
	std::cout << "Testing Neural Network: " << std::endl;

	NeuralNetwork nn;
	NN_Parameters params{ 2,3,2 };

	std::cout << "Testing adding + remove layers: " << std::endl;
	nn.setParameters(params);
	nn.addLayer(std::shared_ptr<FullyConnectedLayer>(), 0); //specifying position
	nn.addLayer(std::shared_ptr<PoolingLayer>(), 1);
	nn.addLayer(std::shared_ptr<ConvolutionLayer>()); //not specifying position: appends
	nn.addLayer(std::shared_ptr<DeconvolutionLayer>());

	std::cout << "Number of layers: (Expect 4)" << std::endl;
	std::cout << nn.getLayers().size() << std::endl; //should be 4

	nn.removeLayer(0); //removes first layer
	std::cout << "Number of layers: (Expect 3)" << std::endl;
	std::cout << nn.getLayers().size() << std::endl; //should be 3


    return;
}

int main() { //Neural Networking Testing 
    testNN();
    testActivation();
    testConvs();

    std::cout << "END TESTING. ENTER ANYTHING TO EXIT" << std::endl;
    char in;
    std::cin >> in;
    return 0;
}
