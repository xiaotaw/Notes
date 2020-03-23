#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "json.hpp"

template <typename T>
bool get_value(const nlohmann::json &j, const std::string & nm, T& val){
    if(j.find(nm) == j.end()) {
        std::cout << nm << " is not found" << std::endl;
        return false;
    }
    else{
        j.at(nm).get_to(val);
        return true;
    }
}

class Data{
private:
    int m_v1;
    std::string m_v3;
    std::vector<int> m_v4;

public:
	bool init_from_file(const std::string & file_name){
		std::ifstream in_f(file_name);
		nlohmann::json j = nlohmann::json::parse(in_f);
		in_f.close();

		bool b1 = get_value<int>(j, "v1", m_v1);
		bool b2 = get_value<std::string>(j, "v3", m_v3);
		bool b3 = get_value< std::vector<int> >(j, "v4", m_v4);

		return b1 && b2 && b3;
	}
};

int main(){
	Data d;
	if(d.init_from_file("test_json.json")){
		std::cout << "success" << std::endl;
	}else{
		std::cout << "failed" << std::endl;
	}
	return 0;
}
