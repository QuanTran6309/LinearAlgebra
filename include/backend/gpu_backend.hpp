#ifndef GPU_BACKEND
#define GPU_BACKEND

#include "backend/backend.hpp"

namespace LinearAlgebra {


class GPU_Backend : public Backend {
public: 

    void* allocate(size_t bytes) override;
    void deallocate(void **ptr) override;

    void add(void *dest, 
            const void *src1, 
            const void *src2, 
            unsigned int numberOfEntries, 
            const Type& type) override;

    void mult(int m, int n, int k,
            const void *src1,
            const void *src2,
            void *dest,
            const Type& type) override;
};




}
#endif