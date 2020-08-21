/**
 * opengl render
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/21 15:17
 */
#pragma once
#include <iostream>
#include "glad/glad.h" // make sure include glad before glfw
#include <GLFW/glfw3.h>

#include "gl_shader.h"

class GLRender
{
public:
    GLFWwindow *window_;
    GLShaderProgram shader_program_;

    GLRender() {}

    bool InitWindow(const char *title = "glfw+glad", int width = 800, int height = 600);
    bool InitShader(const char *vert_shader_source, const char *frag_shader_source);


    void DrawTriangle(const GLuint vao, const int num_vertex);
};
