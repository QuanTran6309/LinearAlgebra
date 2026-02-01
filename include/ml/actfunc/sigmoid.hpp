#ifndef RELU
#define RELU
#include "actfunc.hpp"
namespace LinearAlg {


class Sigmoid : public ActFunc {

public:
    Sigmoid() = default;
    void compute(Tensor& input) override;
};

}
#endif