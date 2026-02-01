#ifndef ACTFUNC
#define ACTFUNC

#include "algebra/tensor.hpp"
namespace LinearAlg {


class ActFunc {
public:
    virtual ~ActFunc() = 0;

    virtual void compute(Tensor& input) = 0;
};


}
#endif

