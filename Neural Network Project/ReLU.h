#pragma once
#include <vector> 


// f(x) = max(0,x)

double reLU(double ip);			//maybe include this within layers since it's not taking up much space

double softmax(std::vector<int> ip);

