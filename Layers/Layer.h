//
// Created by fyatsenko on 12.04.19.
//

#ifndef NNCPP_LAYER_H
#define NNCPP_LAYER_H

class Layer {
protected:
    int outputLength;
    int prevNeuronCount;

    float *values;

    static constexpr float eta = 0.15;
    static constexpr float alfa = 0.5;

public:
    Layer(int neuronsCount, int prevNeuronCount){
        this->outputLength = neuronsCount;
        this->prevNeuronCount = prevNeuronCount;
        this->values = nullptr;
    };
    int getOutputLength() const { return outputLength; };

    virtual ~Layer() { clearValues(); }
    virtual float *feedForward(float * inputs, int batchSize) { this->values = inputs; return inputs; };
    virtual void clearValues() { };
    virtual float *getValues() const { return values; };
};

#endif //NNCPP_LAYER_H
