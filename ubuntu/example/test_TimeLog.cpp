#ifndef __TIME_LOG_H__
#define __TIME_LOG_H__

#include <chrono>
#include <string>

class TimeLogger{
    
    public:
        using Tp = std::chrono::steady_clock::time_point;
        static void printTimeLog(const std::string & message, const std::string pre="");

    private:
        explicit TimeLogger();
        Tp t0, t1, t2;
        double duration0, duration1;
        
        void print(const std::string & message, const std::string pre="");

        static TimeLogger m_static_timer;
        
};
#endif

#include <iomanip>
#include <iostream>
// constructor
TimeLogger::TimeLogger(){
    t0 = t1 = t2 = std::chrono::steady_clock::now();
    duration0 = duration1 = 0.0; 
}

// private print
void TimeLogger::print(const std::string & message, const std::string pre){
    t2 = std::chrono::steady_clock::now();
    duration0 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count();
    duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << pre << std::setw(10) << duration1 << "   "  << std::setw(15) << duration0 / 1e6 << "   " << message << std::endl;
    t1 = t2;
}


// static public print
void TimeLogger::printTimeLog(const std::string& message, const std::string pre){
    m_static_timer.print(message, pre);
}

// static data
TimeLogger TimeLogger::m_static_timer = TimeLogger();



int main(){
    TimeLogger::printTimeLog("1");

    int a;
    std::cin >> a;

    TimeLogger::printTimeLog("2");
    return 0;
}
