#include <iostream>
#include <string>
#include <vector>
#include <cstring>


char* lower2upper(char* str){
    char * p = str;
    while(*p){
        *p += ('A' - 'a');
        p++;
    }
    return str;
}

int main(){
    std::vector<std::string> vec_str;
    vec_str.push_back("abc");
    vec_str.push_back("efg");
    
    std::cout << "vec_str is a vector, containing two strings, " << std::endl;
    std::cout << "    vec_str[0]: " << vec_str[0] << std::endl;
    std::cout << "    vec_str[1]: " << vec_str[1] << std::endl << std::endl;

    std::cout << "get the first item in vec_str, and asign it to str_0, " << std::endl;
    std::string str_0 = vec_str[0];
    std::cout << "    str_0: " << str_0 << std::endl;

    std::cout << std::endl << " ... do something on str_0 ... " << std::endl;
    std::cout << "convert lower to upper inplace by: ";
    std::cout << "lower2upper((char*)str_0.c_str()) " << std::endl << std::endl;;
    // compile error
    // lower2upper(str_0.c_str());
    lower2upper((char*)str_0.c_str());
    
    std::cout << "And Now:" << std::endl;
    std::cout << "    str_0: " << str_0 << std::endl;
    std::cout << "    vec_str[0]: " << vec_str[0] << std::endl << std::endl;

    std::cout << "Anather way: " << std::endl;
    std::string str_1 = vec_str[1];
    char * _str_1 = new char[str_1.size() + 1];
    strcpy(_str_1, str_1.c_str());
    std::string str_1_upper = lower2upper(_str_1);
    std::cout << "    str_1: " << str_1 << std::endl;
    std::cout << "    vec_str[1] " << vec_str[1] << std::endl;
    std::cout << "    str_1_upper: " << str_1_upper << std::endl;

    return 0;
    
}
