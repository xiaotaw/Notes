/**
 * opengl shader program
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/21 12:53
 */
#include <iostream>
#include "gl_shader.h"

// public method

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