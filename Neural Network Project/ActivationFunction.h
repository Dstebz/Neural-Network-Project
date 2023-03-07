#pragma once

// f(x) = max(0,x)


class ActivationFunction {
public:
	
	static double reLU(double ip);			//maybe include this within layers since it's not taking up much space
	static double tanh(double ip);
	static double sigmoid(double ip);	
	static double softmax(double ip);

};