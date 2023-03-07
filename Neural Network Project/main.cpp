#include <Eigen>
#include <iostream>

Eigen::MatrixXi a{      // construct a 2x2 matrix
      {1, 2},     // first row
      {3, 4}      // second row
};

int main() {
	std::cout << a << std::endl;
}