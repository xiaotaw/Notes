/**
 * opengl render 
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/21 15:25
 */
#include <iostream>
#include "gl_render.h"

static void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

static void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
}

bool GLRender::InitWindow(const char *title, int width, int height)
{
    if (!glfwInit())
    {
        std::cout << "[ERROR] init glfw failed " << std::endl;
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window_ = glfwCreateWindow(width, height, title, NULL, NULL);
    if (window_ == nullptr)
    {
        std::cout << "[ERROR] failed to create glfw window" << std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window_);
    glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);

    if (!gladLoadGLLoader((GLADloadproc(glfwGetProcAddress))))
    {
        std::cout << "[ERROR] Failed to initialize GLAD" << std::endl;
        return false;
    }
    return true;
}

bool GLRender::InitShader(const char *vert_shader_source, const char *frag_shader_source)
{
    return shader_program_.Compile(vert_shader_source, frag_shader_source);
}

void GLRender::DrawTriangle(const GLuint vao, int num_vertex)
{
    processInput(window_);
    // render something
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader_program_.UseProgram();
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, num_vertex);

    glfwSwapBuffers(window_);
    glfwPollEvents();
}