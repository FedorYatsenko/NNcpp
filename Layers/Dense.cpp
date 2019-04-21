//
// Created by fyatsenko on 12.04.19.
//
#include <iostream>
#include <random>
#include <ctime>

#include "Dense.h"

Dense::Dense(int neuronsCount, int prevNeuronCount) : Layer(neuronsCount, prevNeuronCount) {
    this->weights = new float[neuronsCount * prevNeuronCount];
    this->deltaWeights = new float[neuronsCount * prevNeuronCount];
    this->bias = new float[neuronsCount];
    this->deltaBias = new float[neuronsCount];

    std::mt19937 gen(time(nullptr));
    std::uniform_real_distribution<float> urd(0.0, 1.0);
//    std::uniform_real_distribution<float> urd(-0.2, 0.2);

    for(int i=0; i < neuronsCount; i++ ) {
        for(int j=0; j < prevNeuronCount; j++) {
            this->weights[i*prevNeuronCount + j] = urd(gen);
            this->deltaWeights[i*prevNeuronCount + j] = 0.0;
        }

        this->bias[i] = 1.0;
        this->deltaBias[i] = 0.0;
    }

    this->values = nullptr;
    this->gradient = nullptr;
}

Dense::~Dense() {
    delete[] weights;
    delete[] deltaWeights;
    delete[] bias;
    delete[] deltaBias;
    clearValues();
}

void Dense::clearValues() {
    delete[] values;
    delete[] gradient;

    values = nullptr;
    gradient = nullptr;
}

float* Dense::feedForward(float *inputs, int inputCount)  {
    // inputCount = batchCount for train
    this->values = new float[inputCount*(this->prevNeuronCount)];

    // inputId = batchId for train
    for(int inputId = 0; inputId < inputCount; inputId++) {
        for(int neuronId = 0; neuronId < (this->outputLength); neuronId++){
            this->values[inputId*(this->prevNeuronCount) + neuronId] = this->bias[neuronId];
//            std::cout << "bias = " << bias[neuronId] << std::endl;

            for(int prevNeuronId = 0; prevNeuronId < (this->prevNeuronCount); prevNeuronId++) {
//                std::cout << " inp: " << inputs[inputId*(this->prevNeuronCount)+prevNeuronId];
//                std::cout << " w: " << this->weights[neuronId*(this->prevNeuronCount) + prevNeuronId];
//                std::cout << " out(t): " << this->values[inputId*(this->prevNeuronCount) + neuronId];
                this->values[inputId*(this->prevNeuronCount) + neuronId] +=
                        inputs[inputId*(this->prevNeuronCount) + prevNeuronId] *
                        this->weights[neuronId*(this->prevNeuronCount) + prevNeuronId];
//                std::cout << " out(t+1): " << this->values[inputId*(this->prevNeuronCount) + neuronId];
//                std::cout<< std::endl;
            }

//            std::cout << "Before tanh: " << outputs[inputId][neuronId];
            this->values[inputId*(this->prevNeuronCount) + neuronId] =
                    this->activation(this->values[inputId*(this->prevNeuronCount) + neuronId]);
//            std::cout << ". After tanh: " << outputs[inputId][neuronId];
//            std::cout<< std::endl;
        }

    }

    return this->getValues();
}

//float** Dense::backPropagation(float **inputs, int batchSize) {
//    return inputs;
//}

void Dense::calcOutputGradients(const float *targetOutputs, int outputCount) {
    this->gradient = new float[this->outputLength];

    float delta;
    for(int neuronId = 0; neuronId < this->outputLength; neuronId++) {
        delta = targetOutputs[neuronId] - this->values[neuronId];
//        std::cout << "Taget:" << targetOutputs[neuronId];
//        std::cout << "   Output:" << this->values[neuronId];
//        std::cout << "   Delta:" << delta;
//        std::cout << "   activationDerivative:" << this->activationDerivative(this->values[neuronId]);

        this->gradient[neuronId] =
                delta * this->activationDerivative(this->values[neuronId]);

//        std::cout << "   Output gradient:" << this->gradient[neuronId] << std::endl;
    }
}

void Dense::calcGradients(const Dense *nextLayer) {
    this->gradient = new float[this->outputLength];
    float sum;

    for (int neuronId = 0; neuronId < this->outputLength; ++neuronId) {
        sum = nextLayer->sumDOW(neuronId);

//        std::cout << "Neuron id: " << neuronId;
//        std::cout << "   sum: " << sum;
//        std::cout << "   activationDerivative: " << Dense::activationDerivative(this->values[0*(this->prevNeuronCount)+ neuronId]) << std::endl;
        this->gradient[neuronId] =
                sum * Dense::activationDerivative(this->values[0*(this->prevNeuronCount)+ neuronId]);
    }
}

void Dense::updateWeights(const Layer *prevLayer) {
    float oldDeltaWeight;
    float newDeltaWeight;

    float prevNeuronValue;

    for (int prevNeuronId = 0; prevNeuronId < this->prevNeuronCount; ++prevNeuronId) {
        prevNeuronValue = (prevLayer->getValues())[0*(this->prevNeuronCount) + prevNeuronId];

        for (int neuronId = 0; neuronId < this->outputLength; ++neuronId) {
            oldDeltaWeight = this->deltaWeights[neuronId*(this->prevNeuronCount) + prevNeuronId];
            newDeltaWeight = eta * prevNeuronValue * this->gradient[neuronId]
                    + alfa * oldDeltaWeight;

            this->deltaWeights[neuronId*(this->prevNeuronCount) + prevNeuronId] = newDeltaWeight;
//            std::cout << "Prev neuron value: " << prevNeuronValue;
//            std::cout << "   Gradient: " << this->gradient[neuronId];
//            std::cout << "   Delta Weight: " << newDeltaWeight << std::endl;
            this->weights[neuronId*(this->prevNeuronCount) + prevNeuronId] += newDeltaWeight;
        }
    }

    for (int neuronId = 0; neuronId < this->outputLength; ++neuronId) {
        oldDeltaWeight = this->deltaBias[neuronId];
        newDeltaWeight = eta * this->bias[neuronId] * this->gradient[neuronId]
                + alfa * oldDeltaWeight;

        this->deltaBias[neuronId] = newDeltaWeight;
        this->bias[neuronId] += newDeltaWeight;
    }
}

float Dense::sumDOW(int prevNeuronId) const {
    float sum = 0.0;

    for (int neuronId = 0; neuronId < this->outputLength; ++neuronId) {
//        std::cout << "Weights: " << weights[neuronId*(this->prevNeuronCount) + prevNeuronId];
//        std::cout << "   Gradient: " << this->gradient[neuronId] << std::endl;
//        std::cout << "sum before: " << sum;

        sum += this->weights[neuronId*(this->prevNeuronCount) + prevNeuronId] * this->gradient[neuronId];
//        std::cout << ",   sum after: " << sum << std::endl;
    }

    return sum;
}

float* Dense::getValues() const {
    return values;
}
