/**
 * @file logging.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2020-08-24
 * @note the code comes from: https://github.com/weigao95/surfelwarp
 * @copyright Copyright (c) 2021
 */
#pragma once
#include "common/disable_copy_assign_move.h"
#include <iostream>
#include <sstream>

#define LOG(severity) LOG_##severity.stream()

#define LOG_INFO LogMessage(__FILE__, __LINE__)
#define LOG_WARNING LOG_INFO
#define LOG_FATAL LogMessageFatal(__FILE__, __LINE__)
#define LOG_ERROR LOG_FATAL
#define LOG_BEFORE_THROW LogMessage().stream()

// The log message
class LogMessage {
public:
  // Constructors
  LogMessage() : log_stream_(std::cout) {}
  LogMessage(const char *file, int line) : log_stream_(std::cout) {
    log_stream_ << file << ":" << line << ": ";
  }
  DISABLE_COPY_ASSIGN(LogMessage);

  // Another line
  ~LogMessage() { log_stream_ << std::endl; }

  std::ostream &stream() { return log_stream_; }

protected:
  std::ostream &log_stream_;
};

class LogMessageFatal {
public:
  LogMessageFatal(const char *file, int line) {
    log_stream_ << file << ":" << line << ": ";
  }

  DISABLE_COPY_ASSIGN(LogMessageFatal);

  // Die the whole system
  ~LogMessageFatal() {
    LOG_BEFORE_THROW << log_stream_.str();
    throw new std::runtime_error(log_stream_.str());
  }

  // The output string stream
  std::ostringstream &stream() { return log_stream_; }

protected:
  std::ostringstream log_stream_;
};