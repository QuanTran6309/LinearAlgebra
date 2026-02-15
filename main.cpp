#include "include/math/tensor.hpp"

#include <iostream>


using namespace LinearAlgebra;
int main(){

    int tensorPtr[] = {
        0, 1, 2, 
        3, 4, 5,

        6, 7, 8,
        9, 10, 11
    };
    Tensor tensor(tensorPtr, {3, 2, 2}, Type::INT, Device::CPU);

    std::cout << tensor.toString() << std::endl;

    return 0;
}