#pragma once

#include <Eigen>

template <class T>

class Layer { //interface class

protected:
public:
	virtual Layer(); //default constructor, pure virtual not required
	
	//virtual Eigen::MatrixXd Run(Eigen::MatrixXd input); //return matrix / image? //Commented out as it causes linker errors
	virtual T getParameters() = 0;
	virtual void setParameters(T params) = 0; 
};
