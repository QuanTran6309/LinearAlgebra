#include "backend/cpu_backend.hpp"
#include <cstdlib>
#include <cstring>
#define NOCUDA

#include "utils/logger.cuh"

namespace LinearAlgebra {

void *CPU_Backend::allocate(size_t bytes) {
    void *ptr = std::malloc(bytes);

    // Handle allocating failure
    if (ptr == nullptr){
        LOGEXCEPTION("Failed to allocate " + std::to_string(bytes) + " on memory");
    }

    return ptr;
}


void CPU_Backend::deallocate(void **ptr){
    std::free(*ptr);
    *ptr = nullptr;
}


void CPU_Backend::copy(void *srcPtr, void *destPtr, size_t bytes){
    std::memcpy(srcPtr, destPtr, bytes);
}



}