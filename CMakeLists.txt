cmake_minimum_required(VERSION 3.13)
project(NNcpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(NNcpp main.cpp Layers/Dense.cpp Layers/Dense.h NeuralNetwork.cpp NeuralNetwork.h Layers/Layer.h)