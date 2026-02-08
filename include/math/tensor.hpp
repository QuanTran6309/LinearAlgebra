#ifndef TENSOR
#define TENSOR

#include "type.hpp"
#include <vector>
#include <string>

namespace LinearAlgebra {


class Tensor {

protected: 
    void *tensorPtr;
    std::vector<unsigned int> dimensions;
    unsigned int totalEntries;
    unsigned int entrySize;      // The size of the TensorType type
    Type tensorType;
    size_t tensorSize;
public: 
    virtual ~Tensor();

    // srcPtr must be on the same location with device
    Tensor(void *srcPtr, const std::vector<unsigned int>& dimensions, Type type, Device device);
    size_t getTensorSize();


    std::string toString();

};



}


#endif