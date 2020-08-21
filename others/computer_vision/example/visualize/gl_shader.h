/**
 * opengl shader program
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/21 12:50
 */
#pragma once
#include "glad/glad.h"

class GLShaderProgram
{
public:
    GLuint shader_program_;

    GLShaderProgram() {}

    ~GLShaderProgram()
    {
        glDeleteProgram(shader_program_);
    }

    GLint Compile(const char *vert_shader_source, const char *frag_shader_source);

    void UseProgram();

private:
    GLint CompileShaderSource(GLuint shader, const char *source);

    GLint LinkProgram(GLuint vert_shader, GLuint frag_shader);
};