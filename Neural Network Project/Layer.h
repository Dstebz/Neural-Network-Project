#pragma once

#include <Eigen>

class BaseLayer { //Interface for layer list
public:
	virtual Eigen::MatrixXd Run(Eigen::MatrixXd input);
	
	virtual ~BaseLayer();
	
};


/*
template <typename T>
class Layer : public BaseLayer { //Templated interface

protected:
public:
	Layer(); 
	virtual ~Layer();
	
	//
	virtual T getParameters() = 0;
	virtual void setParameters(T params) = 0; 
};
*/

