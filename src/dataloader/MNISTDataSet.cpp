#include "MNISTDataSet.h"
#include "MNISTLoader.h"

#include <stdexcept>
#include <algorithm>

#include "logging/Logger.h"

bool MNISTDataSet::LoadTrainingSet(const std::string& imagesPath, const std::string& labelsPath)
{
    // Load the images
    try
    {
        images = MNISTLoader::loadImages(imagesPath);
    }
    catch (const std::exception& e)
    {
        LOG(LogLevel::ERROR, e.what());
        return false;
    }

    // Load Labells
    std::vector<uint8_t> rawLabels;
    try
    {
        rawLabels = MNISTLoader::loadLabels(labelsPath);
    }
    catch (const std::exception& e)
    {
        LOG(LogLevel::ERROR, e.what());
        return false;
    }

    // Check for number mismatch
    if (images.size() != rawLabels.size())
    {
        LOG(LogLevel::ERROR, "number of images do not match the number of labels");
        return false;
    }
    
    labelsOneHot.resize(rawLabels.size(), std::vector<double>(10, 0.0));
    for (size_t i = 0; i < rawLabels.size(); ++i)
    {
        uint8_t label = rawLabels[i];

        if (label >= 10)
        {
            LOG(LogLevel::ERROR, "incalid label for MNIST");
        }
        labelsOneHot[i][label] = 1.0;
    }

    return true;
}

size_t MNISTDataSet::Size() const
{
    return images.size();
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> MNISTDataSet::GetBatch(size_t startIndex, size_t batchSize) const
{
    // out of range check
    if (startIndex >= Size())
    {
        LOG(LogLevel::ERROR, "start index out of range");
        return {{}, {}};
    }

    size_t endIndex = std::min(startIndex + batchSize, Size()); // todo: possibly Size() - 1?
    size_t actualBatchSize = endIndex - startIndex;

    std::vector<std::vector<double>> batchImages(actualBatchSize);
    std::vector<std::vector<double>> batchLabels(actualBatchSize);

    for (size_t i = 0; i < actualBatchSize; ++i)
    {
        batchImages[i] = images[startIndex + i];
        batchLabels[i] = labelsOneHot[startIndex + i];
    }

    return {batchImages, batchLabels};
}
