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
    std::cout << std::endl << "-------" << std::endl;

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
    std::cout << std::endl << "-------" << std::endl;
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

    Eigen::MatrixXd blurKernel = Eigen::MatrixXd::Constant(2, 2, 0.5);
    Eigen::MatrixXd blurTest = Eigen::MatrixXd::Zero(4, 4);
    blurTest << 0,2,0,2,
				2,0,2,0,
				0,2,0,2,
				2,0,2,0;
    

    conv3.setKernel(blurKernel);
    conv3.setParameters({ 1,1,2,0,0 });
    convolved = conv3.Run(blurTest);
    std::cout << convolved << std::endl;

    std::cout << "Expected: " << std::endl;
    Eigen::MatrixXd expected = Eigen::MatrixXd::Constant(5, 5, 0);
    expected << 0,1,1,1,1,
				1,2,2,2,1,
				1,2,2,2,1,
				1,2,2,2,1,
				1,1,1,1,0;
    std::cout << expected << std::endl;

    
    return;

};

//testing deconvolution layer
void testDeconv()
{
    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Testing Deconvolution" << std::endl;

    std::cout << "Testing Parameters" << std::endl;
    DeconvolutionLayer deconv1;
    DeconvolutionLayer deconv2({ 1,1,2,0,0 }) ;
    DeconvolutionLayer deconv3(
        { 1, 1, 4, 0, 0 },
        Eigen::MatrixXd::Constant(4, 4, 2));

    //Testing Constructors
    std::cout << std::endl << "Testing Constructors" << std::endl;

    std::cout << std::endl << "Default:" << std::endl;
    std::cout << "Kernel" << std::endl;
    std::cout << deconv1.getKernel() << std::endl;
    std::cout << "Kernel Size Parameter: " << std::endl;
    std::cout << deconv1.getParameters().kernelSize << std::endl;

    std::cout << std::endl << "Parameters only:" << std::endl;
    std::cout << "Kernel" << std::endl;
    std::cout << deconv2.getKernel() << std::endl;
    std::cout << "Kernel Size Parameter: " << std::endl;
    std::cout << deconv2.getParameters().kernelSize << std::endl;

    std::cout << std::endl << "Parameters and Kernel:" << std::endl;
    std::cout << "Kernel" << std::endl;
    std::cout << deconv3.getKernel() << std::endl;
    std::cout << "Kernel Size Parameter: " << std::endl;
    std::cout << deconv3.getParameters().kernelSize << std::endl;


    //testing setters
    std::cout << std::endl << "Testing Setters" << std::endl;

    DeconvolutionLayer deconv4;
    std::cout << "Before kernel size parameter: " << deconv4.getParameters().kernelSize << std::endl;
    deconv4.setParameters({ 1,1,4,0,0 });
    std::cout << "After size parameter: " << deconv4.getParameters().kernelSize << std::endl;

    std::cout << "Before Kernel: " << std::endl;
    std::cout << deconv4.getKernel() << std::endl;

    Eigen::MatrixXd newKernel = Eigen::MatrixXd::Constant(4, 4, 2.6);
    deconv4.setKernel(newKernel);
    std::cout << "After Kernel:" << std::endl;
    std::cout << deconv4.getKernel() << std::endl;

	//testing run
    std::cout << std::endl << "Testing Run" << std::endl;

	Eigen::MatrixXd testKernel = Eigen::MatrixXd::Constant(2, 2, 1);
    DeconvolutionLayer deconv5({ 1,1,2,0,0 }, testKernel);
	Eigen::MatrixXd testInput = Eigen::MatrixXd::Constant(4, 4, 2);

	std::cout << "Input: " << std::endl;
	std::cout << testInput << std::endl;
	std::cout << std::endl << "Kernel: " << std::endl;
	std::cout << testKernel << std::endl;
	std::cout << "Output:" << std::endl;
	std::cout << deconv5.Run(testInput) << std::endl;

    return;
};

//testing Neuralnetwork object
void testNN()
{
    std::cout << std::endl << "-------" << std::endl;
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
    std::list<std::shared_ptr<BaseLayer>> LayerList = nn.getLayers();

    std::cout << std::endl << "Testing runs!" << std::endl;
    NeuralNetwork nn2;

    std::cout << "Adding One Convolution layer" << std::endl;
    ConvolutionLayer conv1({1,1,3,0,0});
    Eigen::MatrixXd verticalFilter = Eigen::MatrixXd::Zero(3, 3);
    verticalFilter << -1,-1,-1,
					  -1,8,-1,
					  -1,-1,-1;
    conv1.setKernel(verticalFilter);
    nn2.addLayer(std::make_shared<ConvolutionLayer>(conv1));

    Eigen::MatrixXd input = Eigen::MatrixXd::Zero(5, 5);
    input << 10,0,0,0,10,
             10,10,0,0,10,
			 10,0,10,0,10,
			 10,0,0,10,10,
			 10,0,0,0,10;

    std::cout << "Kernel:" << std::endl;
    std::cout << verticalFilter << std::endl;
    std::cout << std::endl << "Input:" << std::endl;
    std::cout << input << std::endl;

    std::cout << std::endl << "NN output:" << std::endl;
    std::cout << nn2.Run(input) << std::endl;

    std::cout << std::endl << "Chaining" << std::endl;
    std::cout << "Same Convolution -> RELU Activation -> avg Pooling" << std::endl;

	ActivationLayer relu(reLU); //activation layer
	PoolingLayer avgPool({3,2,"avg"}); //pooling layer

	NeuralNetwork nn3; //new neural network
    nn3.addLayer(std::make_shared<ConvolutionLayer>(conv1));
	nn3.addLayer(std::make_shared<ActivationLayer>(relu));
    nn3.addLayer(std::make_shared<PoolingLayer>(avgPool));

	nn3.Run(input);

	std::cout << std::endl << "Chained NN output:" << std::endl;
	std::cout << nn3.Run(input) << std::endl;
    
	return;
}

int main() { //Neural Networking Testing 
    
    while(true)
    {
		std::cout << std::endl << "###########" << std::endl;
		std::cout << "Enter a number to test a specific function" << std::endl;
		std::cout << "1: Neural Network" << std::endl;
		std::cout << "2: Activation Layer" << std::endl;
		std::cout << "3: Convolution Layer" << std::endl;
		std::cout << "4: Deconvolution Layer" << std::endl;
		std::cout << "5: Pooling Layer" << std::endl;
		std::cout << "6: Exit" << std::endl;
		std::cout << std::endl << "###########" << std::endl;
		int in;
		std::cin >> in;
		switch (in)
		{
		case 1:
			testNN();
			break;
		case 2:
			testActivation();
			break;
		case 3:
			testConvs();
			break;
		case 4:
			testDeconv();
			break;
		case 5:
			//testPooling();
			break;
		case 6:
			return 0; //break loop
		default:
			std::cout << "Invalid input" << std::endl;
			break;
		}
    }
    
}
