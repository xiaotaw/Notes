/**
 * @file disable_copy_assign_move.h
 * @author xiaotaw (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-04-11
 * @copyright Copyright (c) 2021
 */
#pragma once

#define DISABLE_COPY_ASSIGN_MOVE(T)                                            \
  T(const T &) = delete;                                                       \
  T &operator=(const T &) = delete;                                            \
  T(T &&) = delete;                                                            \
  T &operator=(T &&) = delete

#define DISABLE_COPY_ASSIGN(T)                                                 \
  T(const T &) = delete;                                                       \
  T &operator=(const T &) = delete
