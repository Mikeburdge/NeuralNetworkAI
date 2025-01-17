#include "NeuralNetworkUtility.h"

#include <string>
#include <filesystem>
using namespace std;

std::string NeuralNetworkUtility::GetInitTimestamp()
{
    const chrono::time_point<chrono::system_clock> now = chrono::system_clock::now();

    const time_t nowTime = chrono::system_clock::to_time_t(now);

    tm buffer;
    localtime_s(&buffer, &nowTime);
    
    char formattedTime[100];
    strftime(formattedTime, sizeof(formattedTime), "%Y-%m-%d_%H-%M-%S", &buffer);

    return  formattedTime;
}
