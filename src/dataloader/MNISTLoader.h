#pragma once

#include <vector>
#include <string>

class MNISTLoader {
public:
    /// <summary>
    /// Loads MNIST images from an IDX file.
    /// </summary>
    /// <param name="filePath">Path to the IDX file containing images.</param>
    /// <returns>A 2D vector where each inner vector represents a normalized image.</returns>
    static std::vector<std::vector<double>> loadImages(const std::string& filePath);

    /// <summary>
    /// Loads MNIST labels from an IDX file.
    /// </summary>
    /// <param name="filePath">Path to the IDX file containing labels.</param>
    /// <returns>A vector of labels as integers.</returns>
    static std::vector<uint8_t> loadLabels(const std::string& filePath);

private:
    /// <summary>
    /// Reads 4 bytes from a file and converts them to an integer.
    /// </summary>
    /// <param name="file">Reference to the open file stream.</param>
    /// <returns>The integer value read from the file.</returns>
    static int readInt(std::ifstream& file);
    
};
