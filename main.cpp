#include <iostream>
#include "NeuralNetwork.h"

void trainNN(NeuralNetwork nn) {
    int epochs = 20000;
    int batchSize = 1;

    int samplesCount = 8;

    // {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    // {{1},    {0},    {0},    {0}}; // XOR
    auto *inputs = new float[samplesCount * 2] {0,0, 0,1, 0,0, 0,0, 1,0, 1,1, 0,1, 1,0};
    auto *targetOutputs = new float[samplesCount] {1,0,1,1,0,0,0,0};

    nn.train(inputs, samplesCount, epochs, batchSize, targetOutputs);

    delete[] inputs;
    delete[] targetOutputs;
}

int main() {
    const int layersCount = 3;
    int outputsOnEachLayer[layersCount] = {2, 4, 1};

    NeuralNetwork nn = NeuralNetwork(layersCount, outputsOnEachLayer);

    trainNN(nn);

    std::cout << "End" << std::endl;
    return 0;
}