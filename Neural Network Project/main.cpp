#include <Eigen>
#include <iostream>
#include "ActivationFunction.h"

Eigen::MatrixXd a{      // construct a 2x2 matrix
      {-3,-0.4 },     // first row
      {0.2,2}      // second row
};

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

