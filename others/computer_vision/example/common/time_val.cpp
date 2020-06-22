/**
 * Undocumented
 * @author: xiaotaw
 * @email: 
 * @date: 2020/06/22 09:58
 */
#include <sys/time.h>

#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>

#include "common/time_val.h"


	// ctor
	TimeVal::TimeVal(struct timeval tv){
		sec_ = tv.tv_sec;
		usec_ = tv.tv_usec;
	}
	TimeVal::TimeVal(std::chrono::microseconds us){
		sec_ = us.count() / 1000000;
		usec_ = us.count() % 1000000;
	}
	
	// operator
	TimeVal TimeVal::operator-(const TimeVal tv) const{
		TimeVal tmp;
		tmp.sec_ = sec_ - tv.sec_;
		tmp.usec_ = usec_ - tv.usec_;
		return tmp;
	}

	TimeVal TimeVal::operator+(const TimeVal tv) const{
		TimeVal tmp;
		tmp.sec_ = sec_ +  tv.sec_;
		tmp.usec_ = usec_ + tv.usec_;
		return tmp;
	}

	TimeVal TimeVal::operator/(const int i) const{
		TimeVal tmp;
		tmp.sec_ = sec_ / i;
		tmp.usec_ = usec_ / i;
		return tmp;
	}


	std::string TimeVal::GetTimeStampStr() const{
		std::ostringstream oss;
		oss << *this;
		return oss.str();
	}
	std::string TimeVal::GetTimeStampStr(int precision) const{
		if(precision <= 0 || precision > 6){
			std::cout << "Error!" << std::endl;
			return "";
		}
		std::ostringstream oss;
		oss << sec_ << "." << std::setfill('0') << std::setw(precision) << long(usec_  / pow(10, 6 - precision));
		return oss.str();
	}


	long TimeVal::GetMicroSeconds() const {
		return sec_ * 1000000 + usec_;
	}

	// make usec_ >= 0, it does not care about sec_
	TimeVal TimeVal::Validate(){
		if (usec_ < 0 || usec_ > 1000000){
			long r = usec_ % 1000000;
			long q = usec_ / 1000000;
			if (r < 0){
				q -= 1;
				r += 1000000;
			}
			sec_ += q;
			usec_ = r;
		}
		return *this;
	}


	std::ostream& operator<<(std::ostream &os, const TimeVal tv){
		os << tv.sec_ << "." << std::setfill('0') << std::setw(6) << tv.usec_ << " s";
		return os;
	}
	std::ostringstream& operator<<(std::ostringstream& os, const TimeVal tv){
		os << tv.sec_ << "." << std::setfill('0') << std::setw(6) << tv.usec_;
		return os;
	}

	TimeVal TimeVal::GetCurrentSysTime(){
		struct timeval tv;
		gettimeofday(&tv, nullptr);
		return TimeVal(tv);
	}

	TimeVal TimeVal::Mean(std::vector<TimeVal> tv_vec){
		TimeVal tv_sum;
		for (auto it : tv_vec){
			tv_sum = tv_sum + it;
		}
		return (tv_sum / tv_vec.size()).Validate();
		//TimeVal sum = std::accumulate(tv_vec.begin(), tv_vec.end(), TimeVal());
		//return sum / tv_vec.size();
	}


	TimeVal TimeVal::Std(std::vector<TimeVal> tv_vec, TimeVal tv_mean){
		long sum = 0;
		for (auto it : tv_vec){
			long res = (it - tv_mean).GetMicroSeconds();
			sum += res * res;
			//std::cout << "sum: " << sum << " res: " << res << std::endl;
		}
		long std = long(sqrt(sum / tv_vec.size()));
		return TimeVal(long(std / 1000000), long(std % 1000000));
	}