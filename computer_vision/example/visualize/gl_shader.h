/**
 * opengl shader program
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/21 12:50
 */
#pragma once
#include <string>
#include "glad/glad.h"
#include <glm/glm.hpp>
#include "utils/snippets.h"

class GLShaderProgram
{
public:
    GLuint shader_program_;

    // ctor
    GLShaderProgram() {}
    GLShaderProgram(const std::string vert_shader_filename, const std::string frag_shader_filename);

    // copy assign move
    DISABLE_COPY_ASSIGN(GLShaderProgram);
    GLShaderProgram(GLShaderProgram &&other)
    {
        shader_program_ = other.shader_program_;
        other.shader_program_ = 0;
    }
    GLShaderProgram &operator=(GLShaderProgram &&other)
    {
        if (this != &other)
        {
            shader_program_ = other.shader_program_;
            other.shader_program_ = 0;
        }
        return *this;
    }

    ~GLShaderProgram()
    {

        glDeleteProgram(shader_program_);
    }

    GLint Compile(const char *vert_shader_source, const char *frag_shader_source);

    void UseProgram();

    void SetMat4(const std::string &name, const glm::mat4 &mat);

private:
    GLint CompileShaderSource(GLuint shader, const char *source);

    GLint LinkProgram(GLuint vert_shader, GLuint frag_shader);
};