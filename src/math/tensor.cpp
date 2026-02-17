#include "math/tensor.hpp"
#include "backend/backend.hpp"
#include <numeric>

#define NOCUDA
#include "utils/logger.cuh"

namespace LinearAlgebra {

Tensor::~Tensor(){
    // Determine the location of the tensor first before calling the deallocation
    const Device device = Backend::getPtrDevice(this->tensorPtr);
    Backend *backend = Backend::getBackend(device);
    backend->deallocate(&this->tensorPtr);
}

Tensor::Tensor(void *srcPtr, const std::vector<unsigned int>& dimensions, Type type, Device device)
                : dimensions(dimensions), 
                  totalEntries(std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<unsigned int>())),
                  tensorType(type)
{
    // Keep track of the total size of the tensor in bytes
    this->tensorSize = this->totalEntries * typeSize(this->tensorType);
    if (this->tensorSize <= 0){
        LOGEXCEPTION("Invalid tensor format or type");
    }

    // Get a backend pointer
    Backend *backend = Backend::getBackend(device);

    // Using the backend pointer to allocate a chunk of memory for this tensor
    this->tensorPtr = backend->allocate(this->tensorSize);
    Backend::copy(srcPtr, this->tensorPtr, this->tensorSize);
}

size_t Tensor::getTensorSize(){
    return this->tensorSize;
}


/**
 * toStringHelper()
 * 
 * This function is where the actual tensor string formation happens.
 */
template<typename T>
std::string toStringHelper(T *tensorPtr,
                           const std::vector<unsigned int>& dimensions,
                           unsigned int nth_dim,
                           unsigned int offset,
                           unsigned int depth)
{
    std::string buffer = "";
    if (nth_dim == 0){
        buffer = std::to_string(tensorPtr[offset]);
        for (unsigned int i = 1; i < dimensions[nth_dim]; i++){
            buffer += (", " + std::to_string(tensorPtr[offset + i]));
        }
        return buffer;
    }

    std::vector<std::string> brac_buffer(dimensions[nth_dim]);
    for (unsigned int i = 0; i < brac_buffer.size(); i++){
        // What the fuck?
        brac_buffer[i] = "[" + toStringHelper<T>(tensorPtr, dimensions, nth_dim - 1, i * std::accumulate(dimensions.begin(), dimensions.begin() + nth_dim, 1, std::multiplies<unsigned int>()) + offset, depth + 1) + "]";
    }

    buffer = brac_buffer[0];
    for (unsigned int i = 1; i < brac_buffer.size(); i++){
        buffer += (",\n" + std::string(depth, ' ') + brac_buffer[i]);
    }

    return buffer;
}

// This is just a wrapper for toStringHelper
std::string Tensor::toString(){
    void *bufferPtr = this->tensorPtr;

    // If the current tensor pointer location is in the GPU,
    // must first copy to the CPU first to print
    bool isTensorPtrOnCPU = Backend::getPtrDevice(bufferPtr) == Device::CPU;
    if (!isTensorPtrOnCPU){
        bufferPtr = std::malloc(this->tensorSize);
        Backend::copy(this->tensorPtr, bufferPtr, this->tensorSize);
    }

    std::string buffer;

    // This pattern is used widely across this project to auto detecting the tensor type
    // without having to use switch case everywhere
    auto stringFormationFunctor = [&](auto type_t){
        using T = decltype(type_t);
        buffer = toStringHelper<T>(static_cast<T *>(bufferPtr), this->dimensions, this->dimensions.size() - 1, 0, 0);
    };
    functionTypeDispatcher(this->tensorType, stringFormationFunctor);
   
    if (!isTensorPtrOnCPU){
        std::free(bufferPtr);
    }
    return buffer;
}


}