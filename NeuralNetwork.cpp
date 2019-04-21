//
// Created by fyatsenko on 12.04.19.
//

#include "NeuralNetwork.h"

#include <iostream>
#include <bits/exception.h>
#include <stdexcept>

#include "Layers/Layer.h"
#include "Layers/Dense.h"


NeuralNetwork::NeuralNetwork(int layersCount, int *outputsOnEachLayer) {
    if (layersCount < 2)
        throw std::invalid_argument("Layers count must be at least 2");

    this->layersCount = layersCount;
    layers = new Layer*[layersCount];

    // add Input layer
    layers[0] = new Layer(outputsOnEachLayer[0], outputsOnEachLayer[0]);

    // add hidden layer
    for(int i=1; i<layersCount; i++)
        layers[i] = new Dense(outputsOnEachLayer[i], outputsOnEachLayer[i-1]);
}

NeuralNetwork::~NeuralNetwork() {
    for(int i=0; i<layersCount; i++) {
        delete layers[i];
    }
    delete[] layers;
}

float* NeuralNetwork ::predict(float *inputs, int inputCount) {
    // inputCount = batchCount for train
    for(int i = 0; i < layersCount; i++) {
//        std::cout << "---------- Layer " << i << " -------------" << std::endl;
        inputs = layers[i]->feedForward(inputs, inputCount);
//        layers[i]->clearValues();
    }

    return inputs;
}

void NeuralNetwork ::train(float *inputs, int inputCount, int epochs, int batchSize, float *targetOutputs) {
    int batchCount = inputCount / batchSize;
    int outputNeuronCount = layers[layersCount - 1]->getOutputLength();
    float acc;
    float averageAcc;
    float loss;
    float averageLoss;

    for(int epoch = 0; epoch < epochs; epoch++){
        std::cout << "Epoch: " << epoch + 1 << "/" << epochs << std::endl;
        averageAcc = 0.0;
        averageLoss = 0.0;

        for(int batch = 0; batch < batchCount; batch++) {
            int batchStart = batch * batchSize;
            float *batchInput = &inputs[batchStart * this->layers[0]->getOutputLength()];
            float *batchOutput = NeuralNetwork ::predict(batchInput, batchSize);
            float *batchTargetOutputs = &targetOutputs[batchStart];

            acc = NeuralNetwork::calcAccuracy(batchOutput, batchSize, batchTargetOutputs);
            averageAcc += acc;

            loss = NeuralNetwork::calcLoss(batchOutput, batchSize, batchTargetOutputs);
            averageLoss += loss;

            std::cout << "\tbatch " << batch + 1 << "/" << batchCount;
            std::cout << ":\tacc: " << acc;
            std::cout << ", loss: " << loss;
            std::cout << std::endl;

            auto *outputLayer = (Dense *) this->layers[this->layersCount - 1];
            outputLayer->calcOutputGradients(batchTargetOutputs, batchSize);

            for (int i = this->layersCount - 2; i > 0; i--) {
                ((Dense *) this->layers[i])->calcGradients((Dense *) this->layers[i+1]);
            }

            for (int i = this->layersCount - 1; i > 0; i--) {
                ((Dense *) this->layers[i])->updateWeights((Dense *) this->layers[i-1]);
            }

            for(int i = 0; i < layersCount; i++) {
//                std::cout << "Length: " << layers[i]->getOutputLength() << std::endl;
//                std::cout << "Length: " << ((Dense *) layers[i])->getValues()[0] << std::endl;
                layers[i]->clearValues();
            }
        }
        averageAcc /= batchCount;
        averageLoss /= batchCount;

        std::cout << "Average: acc=" << averageAcc << ", loss=" << averageLoss << std::endl;

        if(averageAcc == 1.0) return;
    }
}

float NeuralNetwork::calcAccuracy(float *outputs, int outputsCount, float *targetOutputs) {
    int correct = 0;

    for(int i = 0; i < outputsCount; i++) {
        std::cout << "\t\toutput[" << i << "] = " << outputs[i];
        std::cout << ", target = " << targetOutputs[i] << std::endl;

        if (abs(outputs[i] - targetOutputs[i]) < 0.01)
            correct += 1;
    }

    return (float)correct / outputsCount;
}

float NeuralNetwork::calcLoss(const float *outputs, int outputsCount, const float *targetOutputs) const {
    // Среднеквадратичная ошибка (RMSE)
    float loss = 0.0;

    int outputLength = this->layers[this->layersCount - 1]->getOutputLength();
    float delta;

    for(int i = 0; i < outputLength; i++) {
        delta = outputs[0*outputLength + i] - targetOutputs[0*outputLength + i];
        loss = delta * delta;
    }

    loss /= outputLength;
    loss = sqrt(loss);

    return loss;
}