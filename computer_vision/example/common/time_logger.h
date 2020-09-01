/**
 * Time Logger, thread-safe not tested
 * @author: xiaotaw
 * @email: 
 * @date: 2020/06/23 15:53
 */
#pragma once
#include <chrono>
#include <string>
#include <mutex>

class TimeLogger{
    
    public:
        using Tp = std::chrono::steady_clock::time_point;
        static void printTimeLog(const std::string & message, const std::string pre="");

    static TimeLogger m_static_timer;
    static std::mutex m_mutex;
    
    private:
        explicit TimeLogger();
        Tp t0, t1, t2;
        double duration0, duration1;
        
        void print(const std::string & message, const std::string pre="");

        
};


