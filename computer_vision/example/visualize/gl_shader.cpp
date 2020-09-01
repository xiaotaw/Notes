/**
 * opengl shader program
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/21 12:53
 */
#include <fstream>
#include <sstream>
#include <iostream>
#include "gl_shader.h"


// TODO: 使用raise/throw抛出异常

static std::string ReadFile(const std::string filename)
{   
    std::ifstream inf(filename, std::ios::in);
    if(!inf.is_open()){
        std::cout << "[ERROR] failed to open file: " << filename << std::endl;
        return std::string("");
    }
    std::stringstream buffer;
    buffer << inf.rdbuf();
    return std::string(buffer.str());
}

// public method
GLShaderProgram::GLShaderProgram(const std::string vert_shader_filename, const std::string frag_shader_filename)
{
    std::string vert_shader_source = ReadFile(vert_shader_filename);
    std::string frag_shader_source = ReadFile(frag_shader_filename);
    Compile(vert_shader_source.c_str(), frag_shader_source.c_str());
}


GLint GLShaderProgram::Compile(const char *vert_shader_source, const char *frag_shader_source)
{
    // compile shader
    GLuint vert_shader_ = glCreateShader(GL_VERTEX_SHADER);
    GLint v_res = CompileShaderSource(vert_shader_, vert_shader_source);
    GLuint frag_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
    GLint f_res = CompileShaderSource(frag_shader_, frag_shader_source);
    // link shader program
    GLint l_res = LinkProgram(vert_shader_, frag_shader_);
    glDeleteShader(vert_shader_);
    glDeleteShader(frag_shader_);
    // maybe a bug
    return v_res && f_res && l_res;
}

void GLShaderProgram::UseProgram()
{
    glUseProgram(shader_program_);
}

// private method

GLint GLShaderProgram::CompileShaderSource(GLuint shader, const char *source)
{
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cout << "[ERROR] failed to compile shader source \n"
                  << infoLog << std::endl;
    }
    return success;
}

GLint GLShaderProgram::LinkProgram(GLuint vert_shader, GLuint frag_shader)
{
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vert_shader);
    glAttachShader(shader_program_, frag_shader);
    glLinkProgram(shader_program_);
    GLint success;
    GLchar infoLog[512];
    glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
    if (!success)
    {
        glGetProgramInfoLog(shader_program_, 512, nullptr, infoLog);
        std::cout << "[ERROR] failed to link shader program \n"
                  << infoLog << std::endl;
    }
    return success;
}

void GLShaderProgram::SetMat4(const std::string &name, const glm::mat4 &mat){
    glUniformMatrix4fv(glGetUniformLocation(shader_program_, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}
