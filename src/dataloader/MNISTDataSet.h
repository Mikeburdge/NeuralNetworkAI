#pragma once

#include <vector>
#include <string>
#include <utility> // for std::pair

///
/// MNISTDataSet is a convenience class that:
///  - Loads MNIST images/labels using MNISTLoader
///  - Converts labels to one-hot vectors
///  - Provides a batch interface for training
///
class MNISTDataSet
{
public:
    /// <summary>
    /// Loads the training set (images + labels) from the specified IDX files.
    /// Internally converts labels to one-hot vectors of size 10.
    /// </summary>
    /// <param name="imagesPath">Path to MNIST train-images-idx3-ubyte</param>
    /// <param name="labelsPath">Path to MNIST train-labels-idx1-ubyte</param>
    /// <returns>True if loaded successfully, false otherwise</returns>
    bool LoadTrainingSet(const std::string& imagesPath, const std::string& labelsPath);

    /// <summary>
    /// Returns total number of samples in the dataset (images.size()).
    /// </summary>
    size_t Size() const;

    /// <summary>
    /// Returns a subset (mini-batch) of images and one-hot labels.
    /// The images are size Nx784, the labels are Nx10 (for digits 0..9).
    /// Indices [startIndex, startIndex + batchSize).
    /// </summary>
    /// <param name="startIndex">Index of the first sample in the batch</param>
    /// <param name="batchSize">How many samples to return</param>
    /// <returns>A pair: (batchImages, batchLabels), each is a std::vector of vectors</returns>
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
        GetBatch(size_t startIndex, size_t batchSize) const;

    /// <summary>
    /// Access the entire set of images (size Nx784).
    /// </summary>
    const std::vector<std::vector<double>>& GetImages() const { return images; }

    /// <summary>
    /// Access the entire set of labels in one-hot form (size Nx10).
    /// </summary>
    const std::vector<std::vector<double>>& GetLabelsOneHot() const { return labelsOneHot; }

private:
    // Stores each training image as a 784 length vector (normalized [0..1]).
    std::vector<std::vector<double>> images;

    // Stores each label as a 10-dimensional one-hot vector. yeah google that one hahah
    std::vector<std::vector<double>> labelsOneHot;
};

