#include "backend/gpu_backend.hpp"
#include <cstdlib>
#include <cstring>
#define NOCUDA

#include "utils/logger.cuh"

namespace LinearAlgebra {

void *GPU_Backend::allocate(size_t bytes) {
    void *ptr = std::malloc(bytes);

    // Handle allocating failure
    if (ptr == nullptr){
        LOGEXCEPTION("Failed to allocate " + std::to_string(bytes) + " on memory");
    }

    return ptr;
}


void GPU_Backend::deallocate(void **ptr){
    std::free(*ptr);
    *ptr = nullptr;
}


void GPU_Backend::copy(void *srcPtr, void *destPtr, size_t bytes){
    std::memcpy(srcPtr, destPtr, bytes);
}


void GPU_Backend::add(void *dest, 
                    const void *src1, 
                    const void *src2, 
                    unsigned int numberOfEntries, 
                    const Type& type)
{

}

void GPU_Backend::mult(int m, int n, int k,
                    const void *src1,
                    const void *src2,
                    void *dest,
                    const Type& type)
{
    
}


}