#include "Logger.h"

#include <filesystem>

using namespace std;

std::string Logger::fileName = "LogFiles/NeuralNetworkAILogFile_" + getInitTimestamp() + ".txt";

void Logger::log(const LogLevel level, const string& message)
{
    if (!filesystem::exists("LogFiles"))
    {
        filesystem::create_directory("LogFiles");
    }

    bool bShouldLog = false;
    
    switch (level)
    {
    case LogLevel::INFO:
        bShouldLog = true;
        break;
    case LogLevel::PROFILING:
        bShouldLog = false;
        break;
    case LogLevel::DEBUG:
        bShouldLog = true;
        break;
    case LogLevel::FLOW:
        bShouldLog = true;
        break;
    case LogLevel::WARNING:
        bShouldLog = true;
        break;
    case LogLevel::ERROR:
        bShouldLog = true;
        break;
    case LogLevel::FATAL:
        bShouldLog = true;
        break;
    }

    if (!bShouldLog)
    {
        return;
    }
    
    ofstream logFile(fileName, ios::app);
    
    string logLevelString = getLogLevelAsString(level);

    string timeStamp = getCurrentTimestamp();

    string logMessage = "[" + timeStamp + "] [" + logLevelString + "] " + message + "\n";

    // log to file and log to console 
    logFile << logMessage;
    cout << logMessage;
}

void Logger::log(const LogLevel level, const string& message, const char* file, const int line, const string& function)
{
    const string combinedMessage = "[" + string(file) + ":" + to_string(line) + " - " + function + "] " + message + "\n";
    log(level, combinedMessage);
}

string Logger::getLogLevelAsString(const LogLevel level)
{
    switch (level)
    {
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::WARNING:
        return "WARNING";
    case LogLevel::ERROR:
        return "ERROR";
    case LogLevel::FATAL:
        return "FATAL";
    default:
        return "UNKNOWN";
    }
}

string Logger::getCurrentTimestamp()
{
    const chrono::time_point<chrono::system_clock> now = chrono::system_clock::now();

    const time_t nowTime = chrono::system_clock::to_time_t(now);

    tm buffer;
    localtime_s(&buffer, &nowTime);
    
    char formattedTime[100];
    strftime(formattedTime, sizeof(formattedTime), "%Y-%m-%d %H:%M:%S", &buffer);

    return  formattedTime;
}

string Logger::getInitTimestamp()
{
    const chrono::time_point<chrono::system_clock> now = chrono::system_clock::now();

    const time_t nowTime = chrono::system_clock::to_time_t(now);

    tm buffer;
    localtime_s(&buffer, &nowTime);
    
    char formattedTime[100];
    strftime(formattedTime, sizeof(formattedTime), "%Y-%m-%d_%H-%M-%S", &buffer);

    return  formattedTime;
}

ScopedLogger::ScopedLogger(string functionName): func(move(functionName))
{
    Logger::log(LogLevel::DEBUG, "Entering " + func);
}

ScopedLogger::~ScopedLogger()
{
    Logger::log(LogLevel::DEBUG, "Exiting " + func);
}

ProfilingLogger::ProfilingLogger(string functionName): func(move(functionName)), start(chrono::high_resolution_clock::now())
{
    Logger::log(LogLevel::PROFILING, "Entering " + func);
}

ProfilingLogger::~ProfilingLogger()
{
    const auto end = chrono::high_resolution_clock::now();
    const auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    Logger::log(LogLevel::PROFILING, "Exiting " + func + " (" + to_string(duration.count()) + " ms)");
}