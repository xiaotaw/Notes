/**
 * opengl render
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/21 15:17
 */
#pragma once
#include <vector>
#include <iostream>

#include "glad/glad.h" // make sure include glad before glfw
#include <GLFW/glfw3.h>
#include <Eigen/Core>

#include "gl_shader.h"
#include "utils/snippets.h"

class GLRender
{
public:
    GLFWwindow *window_;
    GLShaderProgram shader_program_;
    GLuint vao_;
    GLuint point_vbo_;
    GLuint color_vbo_;
    size_t point_buffer_size_;
    size_t color_buffer_size_;
    unsigned num_points_;

    // ctor
    GLRender() {}
    GLRender(const std::string vert_shader_filename, const std::string frag_shader_filename)
    {
        InitWindow();
        shader_program_ = GLShaderProgram(vert_shader_filename, frag_shader_filename);
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &point_vbo_);
        glGenBuffers(1, &color_vbo_);
    }
    DISABLE_COPY_ASSIGN_MOVE(GLRender);
    // dtor
    ~GLRender()
    {
        glDeleteVertexArrays(1, &vao_);
        glDeleteBuffers(1, &point_vbo_);
        glDeleteBuffers(1, &color_vbo_);
        glfwTerminate();
    }

    using Vector3f = Eigen::Vector3f;
    bool InitWindow(const char *title = "glfw+glad", int width = 800, int height = 600);
    // update point cloud
    GLenum UpdatePointCloud(const std::vector<Vector3f> &vertexs, const std::vector<Vector3f> &colors);
    // render point cloud
    GLenum RenderPointCloud();


    // just for debug test
    bool InitShader(const char *vert_shader_source, const char *frag_shader_source);
    void DrawTriangle(const GLuint vao, const int num_vertex);
    void DrawPoint(const GLuint vao, const int num_vertex);
    void DrawTest(const GLuint vao, const int num_vertex);
};
