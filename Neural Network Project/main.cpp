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
    std::cout << "Stride: " << conv1.getParameters().stride << std::endl;
    std::cout << "Padding: " << conv1.getParameters().padding << std::endl;
    std::cout << "Kernel size: " << conv1.getParameters().kernelSize << std::endl;
    std::cout << "Expect: 1,1,3" << std::endl << std::endl;

    std::cout << "conv2 parameters:" << std::endl;
    std::cout << "Stride: " << conv2.getParameters().stride << std::endl;
    std::cout << "Padding: " << conv2.getParameters().padding << std::endl;
    std::cout << "Kernel size: " << conv2.getParameters().kernelSize << std::endl;
    std::cout << "Expect: 1,2,3" << std::endl << std::endl;

    std::cout << "Testing Convolution Kernel: " << std::endl;
    ConvolutionLayer conv3;
    std::cout << "Default Kernel: " << std::endl;
    std::cout << conv3.getKernel() << std::endl;
    std::cout << "Expect: 3x3 matrix of 1s" << std::endl << std::endl;

    Eigen::MatrixXd testKernel = Eigen::MatrixXd::Zero(3, 3);
    testKernel <<
        0, 0, 0,
        0, 1, 0,
        0, 0, 0; //Identity kernel, does not change input

    conv3.setKernel(testKernel);

    std::cout << "Test kernel: " << std::endl;
    std::cout << conv3.getKernel() << std::endl;
    std::cout << "Expect 3x3, 0s on edge, 1 in centre" << std::endl << std::endl;

    std::cout << "Testing Convolution Run():" << std::endl;

    Eigen::MatrixXd testInput = Eigen::MatrixXd::Constant(4, 4, 2);
    conv3.setParameters({ 1,1,3,0,0 });
    Eigen::MatrixXd convolved = conv3.Run(testInput);

    std::cout << convolved << std::endl;
    std::cout << "Expect 4x4 of 2s" << std::endl << std::endl;

	std::cout << "Blur kernel test: " << std::endl;
    Eigen::MatrixXd blurKernel = Eigen::MatrixXd::Constant(2, 2, 0.5);
    Eigen::MatrixXd blurTest = Eigen::MatrixXd::Zero(4, 4);
    blurTest << 0,2,0,2,
				2,0,2,0,
				0,2,0,2,
				2,0,2,0;

	std::cout << "Kernel: " << std::endl;
    std::cout << blurKernel << std::endl;
	std::cout << "Expect 2x2 of 0.5s" << std::endl << std::endl;

	std::cout << "Input: " << std::endl;
	std::cout << blurTest << std::endl;
	std::cout << "Expect 4x4 of 2s and 0s" << std::endl << std::endl;


    conv3.setKernel(blurKernel);
    conv3.setParameters({ 1,1,2,0,0 });
    convolved = conv3.Run(blurTest);
	std::cout << "Output: " << std::endl;
    std::cout << convolved << std::endl;

    std::cout << "Expected: " << std::endl;
    Eigen::MatrixXd expected = Eigen::MatrixXd::Constant(5, 5, 0);
    expected << 0,1,1,1,1,
				1,2,2,2,1,
				1,2,2,2,1,
				1,2,2,2,1,
				1,1,1,1,0;
    std::cout << expected << std::endl << std::endl;

    
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

//testing Fully Connected Layer
void testFC() {

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Testing Fully Connected Layer" << std::endl;

    std::cout << "Testing Parameters" << std::endl;

    //creating 2 objects, 1 default, 1 with selected parameters
    FullyConnectedLayer fc1;
    FullyConnectedLayer fc2({ 2,2 });        //will need to change the weights. Make a new variable called weight_val or something       
    //add a new constructor with third parameter for weight elements

    //testing these constructors
    std::cout << std::endl << "Default Constructor:" << std::endl;
    std::cout << std::endl << "Input Channels" << std::endl;
    std::cout << fc1.getParameters().inputChannels << std::endl;
    std::cout << std::endl << "Output Channels" << std::endl;
    std::cout << fc1.getParameters().outputChannels << std::endl;
    std::cout << std::endl << "Weights" << std::endl;
    std::cout << fc1.getParameters().weights << std::endl;

    std::cout << std::endl << "Parameter Constructor:" << std::endl;
    std::cout << std::endl << "Input Channels" << std::endl;
    std::cout << fc2.getParameters().inputChannels << std::endl;
    std::cout << std::endl << "Output Channels" << std::endl;
    std::cout << fc2.getParameters().outputChannels << std::endl;
    std::cout << std::endl << "Weights" << std::endl;
    std::cout << fc2.getParameters().weights << std::endl;

    //testing setters
    std::cout << std::endl << "Testing Setters" << std::endl;

    FullyConnectedLayer fc4;    //create object
    std::cout << "Before input channels: " << fc4.getParameters().inputChannels << std::endl;
    std::cout << "Before output channels: " << fc4.getParameters().outputChannels << std::endl;
    fc4.setParameters({ 2,3 });
    std::cout << "After input channels: " << fc4.getParameters().inputChannels << std::endl;
    std::cout << "After output channels: " << fc4.getParameters().outputChannels << std::endl;
    
    /*
    //testing run()
    std::cout << std::endl << "Testing Run" << std::endl;

    FullyConnectedLayer fc5;

    Eigen::MatrixXd ip{ {1,2,3,4},
                        {5,6,7,8},
                        {9,10,11,12},
                        {13,14,15,16} };
    Eigen::MatrixXd op;

    op = fc5.Run(ip);                   //note: doesn't work yet because dimensions of ip matrix is not the same as weights. Need to change.
    std::cout << "After Run: " << op << std::endl;
    */
    return;
}

void testPool() {

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Testing the Pooling Layer" << std::endl;

    std::cout << "Testing Parameters" << std::endl;

    //creating 3 objects. 1 default, 1 with set parameters and default pooling, 1 with set parameters and set pooling type
    PoolingLayer pool1;
    PoolingLayer pool2({ 2,2 });
    PoolingLayer pool3({ 3,3, "min" });

    //testing these constructors
    std::cout << std::endl << "Default Constructor:" << std::endl;
    std::cout << std::endl << "Filter Size (note this is square)" << std::endl;
    std::cout << pool1.getParameters().filter << std::endl;
    std::cout << std::endl << "Stride" << std::endl;
    std::cout << pool1.getParameters().stride << std::endl;
    std::cout << std::endl << "Pooling Type" << std::endl;
    std::cout << pool1.getParameters().pooling_type << std::endl;

    std::cout << std::endl << "Parameter Constructor:" << std::endl;
    std::cout << std::endl << "Filter Size (note this is square)" << std::endl;
    std::cout << pool2.getParameters().filter << std::endl;
    std::cout << std::endl << "Stride" << std::endl;
    std::cout << pool2.getParameters().stride << std::endl;
    std::cout << std::endl << "Pooling Type" << std::endl;
    std::cout << pool2.getParameters().pooling_type << std::endl;

    std::cout << std::endl << "Parameter and Pooling Type Constructor:" << std::endl;
    std::cout << std::endl << "Filter Size " << std::endl;
    std::cout << pool3.getParameters().filter << std::endl;
    std::cout << std::endl << "Stride" << std::endl;
    std::cout << pool3.getParameters().stride << std::endl;
    std::cout << std::endl << "Pooling Type" << std::endl;
    std::cout << pool3.getParameters().pooling_type << std::endl;

    //testing setters
    std::cout << std::endl << "Testing Setters" << std::endl;

    PoolingLayer pool4;    //create object
    std::cout << "Before filter size: " << pool4.getParameters().filter << std::endl;
    std::cout << "Before stride: " << pool4.getParameters().stride << std::endl;
    std::cout << "Before pooling type: " << pool4.getParameters().pooling_type << std::endl;
    pool4.setParameters({ 2, 3, "max" });
    std::cout << "After filter size: " << pool4.getParameters().filter << std::endl;
    std::cout << "After stride: " << pool4.getParameters().stride << std::endl;
    std::cout << "After pooling type: " << pool4.getParameters().pooling_type << std::endl;

    //testing pooling methods
    std::cout << std::endl << "Testing Pooling Methods" << std::endl;

    //create input and output matrix
    Eigen::MatrixXd ip{ {1,2,3,4}, 
                        {5,6,7,8},
                        {9,10,11,12},
                        {13,14,15,16}};
    Eigen::MatrixXd op1, op2;
    double op3, op4, op5;

    std::cout << std::endl << "Test input matrix is" << std::endl << ip << std::endl;

    PoolingLayer pool5;

    op1 = pool5.avg_pool(ip,2,2);
    std::cout << std::endl << "Average Pooling with a stride: 2 and filter: 2 " << std::endl << op1 << std::endl;

    op2 = pool5.max_pool(ip, 2, 2);
    std::cout << std::endl << "Max Pooling with a stride: 2 and filter: 2 " << std::endl << op2 << std::endl;

    op3 = pool5.global_pool(ip, "max");
    std::cout << std::endl << "Global Max Pooling " << std::endl << op3 << std::endl;

    op4 = pool5.global_pool(ip, "avg");
    std::cout << std::endl << "Global Average Pooling " << std::endl << op4 << std::endl;

    op5 = pool5.global_pool(ip, "min");
    std::cout << std::endl << "Global Min Pooling " << std::endl << op5 << std::endl;

    //testing run()
    std::cout << std::endl << "Testing Run" << std::endl;

    PoolingLayer pool6;
    Eigen::MatrixXd ip2{ {1,2,3,4},
    						{5,6,7,8},
    						{9,10,11,12},
    						{13,14,15,16} };
    Eigen::MatrixXd op6;

    std::cout << std::endl << "Test input matrix is" << std::endl << ip2 << std::endl;
    op6 = pool6.Run(ip2);
    std::cout << std::endl << "Output matrix is" << std::endl << op6 << std::endl;
    return;
}


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
    std::cout << "Conv -> RELU only" << std::endl;
    std::cout << nn3.Run(input) << std::endl;

    nn3.addLayer(std::make_shared<PoolingLayer>(avgPool));
	
	std::cout << std::endl << "All 3 Chained NN output:" << std::endl;
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
