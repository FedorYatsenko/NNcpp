//
// Created by fyatsenko on 12.04.19.
//

#ifndef NNCPP_NEURALNETWORK_H
#define NNCPP_NEURALNETWORK_H


#include "Layers/Layer.h"

class NeuralNetwork {
private:
    int layersCount;
//    Layer **layers;
    Layer **layers;

    static float calcAccuracy(float *outputs, int outputsCount, float *targetOutputs);
    float calcLoss(const float *outputs, int outputsCount, const float *targetOutputs) const;

public:
    NeuralNetwork(int layersCount, int *outputsOnEachLayer);
    ~NeuralNetwork();
    float *predict(float *inputs, int inputCount);
    void train(float *inputs, int inputCount, int epochs, int batchSize, float *targetOutputs);
};


#endif //NNCPP_NEURALNETWORK_H
