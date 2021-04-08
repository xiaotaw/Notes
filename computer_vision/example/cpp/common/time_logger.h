/**
 * @file time_logger.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2020-06-23
 * @note Time Logger, thread-safe not tested
 * @copyright Copyright (c) 2021
 */
#pragma once
#include <chrono>
#include <mutex>
#include <string>

class TimeLogger {

public:
  using Tp = std::chrono::steady_clock::time_point;
  static void printTimeLog(const std::string &message,
                           const std::string pre = "");

  static TimeLogger m_static_timer;
  static std::mutex m_mutex;

private:
  explicit TimeLogger();
  Tp t0, t1, t2;
  double duration0, duration1;

  void print(const std::string &message, const std::string pre = "");
};
