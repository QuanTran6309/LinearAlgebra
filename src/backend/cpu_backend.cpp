#include "backend/cpu_backend.hpp"
#include <cstdlib>

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

void CPU_Backend::add(void *dest, 
                    const void *src1, 
                    const void *src2, 
                    unsigned int numberOfEntries, 
                    const Type& type)
{

}

void CPU_Backend::mult(int m, int n, int k,
                    const void *src1,
                    const void *src2,
                    void *dest,
                    const Type& type)
{
    
}

}