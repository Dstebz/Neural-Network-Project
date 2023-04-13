#pragma once

#include <Eigen>




class BaseLayer { //Interface for layer list
public:
	virtual Eigen::MatrixXd Run(Eigen::MatrixXd input); //return matrix / image? //Commented out as it causes linker errors
};

template <typename T>
class Layer : BaseLayer { //Templated interface

protected:
public:
	Layer(); //default constructor, pure virtual breaks this?
	
	//
	virtual T getParameters() = 0;
	virtual void setParameters(T params) = 0; 
};
