/**
 * @file time_val.h
 * @author xiaotaw (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2020-06-22
 * @note Only unix support now
 * @copyright Copyright (c) 2021
 */
#pragma once
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sys/time.h>
#include <vector>

/**
 * \brief simple wrap of `struc timeval`
 */
class TimeVal {
public:
  long sec_;
  long usec_;

  // ctor
  TimeVal() : sec_(0), usec_(0) {}
  TimeVal(long sec, long usec) : sec_(sec), usec_(usec) {}
  TimeVal(struct timeval tv);
  TimeVal(std::chrono::microseconds us);
  // dtor
  ~TimeVal() = default;

  // operator
  TimeVal operator-(const TimeVal tv) const;

  TimeVal operator+(const TimeVal tv) const;

  TimeVal operator/(const int i) const;

  // convert into string
  std::string GetTimeStampStr() const;
  std::string GetTimeStampStr(int precision) const;

  // convert into microseconds
  long GetMicroSeconds() const;

  // make usec_ >= 0, it does not care about sec_
  TimeVal Validate();

  // stream
  friend std::ostream &operator<<(std::ostream &os, const TimeVal tv);
  friend std::ostringstream &operator<<(std::ostringstream &os,
                                        const TimeVal tv);

  // current system time
  static TimeVal GetCurrentSysTime();

  static TimeVal Mean(std::vector<TimeVal> tv_vec);
  static TimeVal Std(std::vector<TimeVal> tv_vec, TimeVal tv_mean);
};
