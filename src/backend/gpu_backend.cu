#include "backend/gpu_backend.hpp"
#include <cstdlib>
#include <cstring>

#include "utils/logger.cuh"


namespace LinearAlgebra {

void *GPU_Backend::allocate(size_t bytes) {
    void *ptr = nullptr;
    CUDA_ERR_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}


void GPU_Backend::deallocate(void **ptr){
    CUDA_ERR_CHECK(cudaFree(*ptr));
    *ptr = nullptr;
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