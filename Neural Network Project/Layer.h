#pragma once

#include <string>

class Layer { //interface class

protected:
	virtual struct Parameters {};
public:
	Layer();
	Layer(Parameters);
	~Layer();
	
	virtual void Run(); //return matrix / image?
	virtual Parameters getParameters();
	virtual void setParameters(Parameters); 

};

Layer::Layer() {
	//Empty / default constructor

}

Layer::~Layer() {
	//Destructor
}
