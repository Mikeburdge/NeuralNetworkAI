#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <mutex>

#define LOG(level, message) Logger::log(level, message, __FILE__, __LINE__, __func__)
#define SCOPE_LOG ScopedLogger scopedLog(__FUNCTION__)
#define PROFILE_LOG ProfilingLogger profilingLog(__FUNCTION__)

enum class LogLevel { INFO, DEBUG, WARNING, ERROR, FATAL };

class Logger
{
public:
    
    static void log(LogLevel level, const std::string& message);
    static void log(LogLevel level, const std::string& message, const char* file, int line, const std::string& function);
    
private:
    static std::string getLogLevelAsString(LogLevel level);

    static std::string getCurrentTimestamp();
    
    static std::string getInitTimestamp();

    static std::string fileName;
};


class ScopedLogger
{
public:
    ScopedLogger(std::string functionName);

    ~ScopedLogger();

private:
    std::string func;
};

class ProfilingLogger
{
public:
    ProfilingLogger(std::string functionName);

    ~ProfilingLogger();

private:
    std::string func;
    std::chrono::high_resolution_clock::time_point start;
};
