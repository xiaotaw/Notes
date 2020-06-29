/**
 * Only tested on linux
 * @author: xiaotaw
 * @email: 
 * @date: 2020/06/22 15:07
 */
#ifdef __WIN32
#include <io.h>     /* for access */
#include <direct.h> /* for mkdir */
#else
#include <unistd.h>    /* for access */
#include <sys/types.h> /* for mkdir */
#include <sys/stat.h>  /* for mkdir */
#endif

#include <string>
#include <iostream>
#include <fstream>

inline void CreateDirOrExit(const std::string path)
{
    if (access(path.c_str(), F_OK) < 0)
    {
        int status = 0;
#ifdef __WIN32
        status = mkdir(path.c_str());
#else
        //status = mkdir(path, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        status = mkdir(path.c_str(), S_IRWXU);
#endif
        if(status < 0){
            std::cout << "Error: failed to create dir: " << path << std::endl;
            exit(EXIT_FAILURE);
        }
    }else{
        std::cout << "Info: already exists: " << path << std::endl;
    }
}


inline void OpenFileOrExit(std::ofstream& of, const std::string path, std::ios_base::openmode mode){
    of.open(path, mode);
    if(!of.is_open()){
        std::cout << "Error: failed to create file " << path << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void OpenFileOrExit(std::ofstream& of, const std::string path){
    of.open(path);
    if(!of.is_open()){
        std::cout << "Error: failed to create file " << path << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void OpenFileOrExit(std::ifstream& inf, const std::string path, std::ios_base::openmode mode){
    inf.open(path, mode);
    if(!inf.is_open()){
        std::cout << "Error: failed to open file " << path << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void OpenFileOrExit(std::ifstream& inf, const std::string path){
    inf.open(path);
    if(!inf.is_open()){
        std::cout << "Error: failed to open file " << path << std::endl;
        exit(EXIT_FAILURE);
    }
}
