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

std::string NeuralNetworkUtility::GetTimeStampWithAnnotations()
{
    const chrono::time_point<chrono::system_clock> now = chrono::system_clock::now();

    const time_t nowTime = chrono::system_clock::to_time_t(now);

    tm buffer;
    localtime_s(&buffer, &nowTime);
    
    char formattedTime[100];
    strftime(formattedTime, sizeof(formattedTime), "%YY-%mM-%dD_%HH-%MM-%SS", &buffer);
    
    return  formattedTime;
}

std::string NeuralNetworkUtility::FormatTimeHMS(double secondsTotal)
{
    int totalSec = (int)secondsTotal;
    int hrs  = totalSec / 3600;
    int rem  = totalSec % 3600;
    int mins = rem / 60;
    int secs = rem % 60;

    char buf[64];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d", hrs, mins, secs);
    return std::string(buf);
}