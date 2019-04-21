//
// Created by fyatsenko on 12.04.19.
//

#ifndef NNCPP_DENSE_H
#define NNCPP_DENSE_H

#include <math.h>

#include "Layer.h"

class Dense : public Layer {
protected:
    float *weights;
    float *deltaWeights;

    float *bias;
    float *deltaBias;

    float *gradient;

public:
    Dense(int neuronsCount, int inputLength);
    ~Dense() override;
    float *feedForward(float *inputs, int inputCount) override;

    float activation(float x) const { return tanh(x); };
    float activationDerivative(float x) const { return 1.0 - x * x; };
    void calcOutputGradients(const float *targetOutputs, int outputCount);
    void calcGradients(const Dense *nextLayer);
    void updateWeights(const Layer *prevLayer);
    float sumDOW(int prevNeuronId) const ;
    float *getValues() const override;

    void clearValues() override;
};


#endif //NNCPP_DENSE_H
