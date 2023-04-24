
#include <iostream>

#include "ActivationFunction.h"

#include "Convolution.h"
#include "Deconvolution.h"
#include "FullyConnected.h"
#include "Pooling.h"
#include "NeuralNetwork.h"

/*
Eigen::MatrixXd a{      // construct a 2x2 matrix
      {-3,-0.4 },     // first row
      {0.2,2}      // second row
};
*/
/*
int main() {
	std::cout << a << std::endl;

    std::cout << "testing activation functions: " << std::endl;

    std::cout << "sigmoid function: " << std::endl;

    std::cout << "\n" << "\n";

    ActivationFunction test;

    Eigen::MatrixXd ans1, ans2, ans3, ans4;

    
    ans1 = test.reLU(a);
    std::cout << "after applying the reLU function: " << std::endl << ans1 << std::endl;
    std::cout << "\n" << "\n";
    

    ans2 = test.tanh(a);
    std::cout << "After applying the tanh function: " << std::endl << ans2 << std::endl;
    std::cout << "\n" << "\n";

    ans3 = test.sigmoid(a);
    std::cout << "After applying the sigmoid function: " << std::endl << ans3 << std::endl;
    std::cout << "\n" << "\n";

    
    ans4 = test.softmax(a);
    std::cout << "After applying the softmax function: " << std::endl << ans4 << std::endl;
    std::cout << "\n" << "\n";
    


    return 0;
}
*/


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


};

void testDeconv()
{
    DeconvolutionLayer deconv1;

};

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



}

int main() { //Neural Networking Testing 
    testNN();
    while (true) {}; //loop nothing, keeps window open
}
