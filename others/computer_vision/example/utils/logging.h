/**
 * Logging. 
 *  Note: the code comes from: https://github.com/weigao95/surfelwarp
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/24 18:00
 */
#pragma once
#include <iostream>
#include <sstream>

#include "snippets.h"

#define LOG(severity) LOG_##severity.stream()

#define LOG_INFO LogMessage(__FILE__, __LINE__)
#define LOG_WARNING LOG_INFO
#define LOG_FATAL LogMessageFatal(__FILE__, __LINE__)
#define LOG_ERROR LOG_FATAL
#define LOG_BEFORE_THROW LogMessage().stream()

// The log message
class LogMessage {
public:
    //Constructors
    LogMessage() : log_stream_(std::cout) {}
    LogMessage(const char* file, int line) : log_stream_(std::cout) {
        log_stream_ << file << ":" << line << ": ";
    }
    DISABLE_COPY_ASSIGN(LogMessage);
    
    //Another line
    ~LogMessage() { log_stream_ << "\n"; }
    
    std::ostream& stream() { return log_stream_; }
protected:
    std::ostream& log_stream_;
};

class LogMessageFatal {
public:
    LogMessageFatal(const char* file, int line) {
        log_stream_ << file << ":" << line << ": ";
    }
    
    DISABLE_COPY_ASSIGN(LogMessageFatal);
    
    //Die the whole system
    ~LogMessageFatal() {
        LOG_BEFORE_THROW << log_stream_.str();
        throw new std::runtime_error(log_stream_.str());
    }
    
    //The output string stream
    std::ostringstream& stream() { return log_stream_; }
protected:
    std::ostringstream log_stream_;
};