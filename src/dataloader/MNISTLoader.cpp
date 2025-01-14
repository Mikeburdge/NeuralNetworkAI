#include "MNISTLoader.h"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <filesystem>
#include <iostream>

int MNISTLoader::readInt(std::ifstream& file) {
    unsigned char buffer[4];
    file.read(reinterpret_cast<char*>(buffer), 4);
    return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
}

std::vector<std::vector<double>> MNISTLoader::loadImages(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filePath);
    }

    int magicNumber = readInt(file);
    if (magicNumber != 2051) { // 0x00000803
        throw std::runtime_error("Invalid magic number in image file: " + std::to_string(magicNumber));
    }

    int numImages = readInt(file);
    int numRows = readInt(file);
    int numCols = readInt(file);

    int imageSize = numRows * numCols;
    std::vector<std::vector<double>> images(numImages, std::vector<double>(imageSize));

    for (int i = 0; i < numImages; ++i) {
        std::vector<uint8_t> buffer(imageSize);
        file.read(reinterpret_cast<char*>(buffer.data()), imageSize);
        for (int j = 0; j < imageSize; ++j) {
            images[i][j] = buffer[j] / 255.0; // Normalize pixel values to [0, 1]
        }
    }

    file.close();
    return images;
}

std::vector<uint8_t> MNISTLoader::loadLabels(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filePath);
    }

    int magicNumber = readInt(file);
    if (magicNumber != 2049) { // 0x00000801
        throw std::runtime_error("Invalid magic number in label file: " + std::to_string(magicNumber));
    }

    int numLabels = readInt(file);
    std::vector<uint8_t> labels(numLabels);

    file.read(reinterpret_cast<char*>(labels.data()), numLabels);
    file.close();
    return labels;
}