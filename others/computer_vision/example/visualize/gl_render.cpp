/**
 * opengl render. 
 * References: 
 *   https://github.com/microsoft/Azure-Kinect-Sensor-SDK
 *   https://github.com/forestsen/KinectAzureDKProgramming
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/21 15:25
 */
#include <iostream>
#include "gl_render.h"

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
// camera
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
// timing
float deltaTime = 0.0f; // time between current frame and last frame
float lastFrame = 0.0f;

GLCamera GLRender::camera_ = GLCamera(glm::vec3(0.0f, 0.0f, 3.0f));

void GLRender::ProcessInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera_.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera_.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera_.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera_.ProcessKeyboard(RIGHT, deltaTime);
}

void GLRender::framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void GLRender::mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        if (firstMouse)
        {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }
        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;

        lastX = xpos;
        lastY = ypos;

        camera_.ProcessMouseMovement(xoffset, yoffset);
    }
}

void GLRender::scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
{
    camera_.ProcessMouseScroll(yoffset);
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
    glfwSetCursorPosCallback(window_, mouse_callback);
    glfwSetScrollCallback(window_, scroll_callback);

    //glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

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

GLenum GLRender::UpdatePointCloud(const std::vector<Vector3f> &points, const std::vector<Vector3f> &colors)
{

    glBindVertexArray(vao_);

    // transfer points_data(vertexes)
    // check if the size changed
    glBindBuffer(GL_ARRAY_BUFFER, point_vbo_);
    size_t point_data_size = points.size() * sizeof(Vector3f);
    if (point_buffer_size_ != point_data_size)
    {
        num_points_ = points.size();
        point_buffer_size_ = point_data_size;
        glBufferData(GL_ARRAY_BUFFER, point_buffer_size_, nullptr, GL_STREAM_DRAW);
    }
    // copy data into buffer
    GLubyte *point_mapped_buffer = reinterpret_cast<GLubyte *>(glMapBufferRange(GL_ARRAY_BUFFER, 0, point_buffer_size_, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
    if (!point_mapped_buffer)
    {
        return glGetError();
    }
    const GLubyte *point_src = reinterpret_cast<const GLubyte *>(points.data());
    std::copy(point_src, point_src + point_buffer_size_, point_mapped_buffer);
    if (!glUnmapBuffer(GL_ARRAY_BUFFER))
    {
        return glGetError();
    }
    //
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

    // transfer color_data in the same way
    glBindBuffer(1, color_vbo_);
    size_t color_data_size = colors.size() * sizeof(Vector3f);
    if (color_buffer_size_ != color_data_size)
    {
        color_buffer_size_ = color_data_size;
        glBufferData(GL_ARRAY_BUFFER, color_buffer_size_, nullptr, GL_STREAM_DRAW);
    }
    GLubyte *color_mapped_buffer = reinterpret_cast<GLubyte *>(glMapBufferRange(GL_ARRAY_BUFFER, 0, color_buffer_size_, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT));
    if (!color_mapped_buffer)
    {
        return glGetError();
    }
    const GLubyte *color_src = reinterpret_cast<const GLubyte *>(colors.data());
    std::copy(color_src, color_src + color_buffer_size_, color_mapped_buffer);
    if (!glUnmapBuffer(GL_ARRAY_BUFFER))
    {
        return glGetError();
    }
    //
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

    shader_program_.UseProgram();

    glBindVertexArray(0);

    return glGetError();
}

GLenum GLRender::RenderPointCloud()
{
    glUseProgram(shader_program_.shader_program_);

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, num_points_);

    return glGetError();
}

// debug method
void GLRender::DrawTriangle(const GLuint vao, int num_vertex)
{
    ProcessInput(window_);
    glfwPollEvents();
    // render something
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader_program_.UseProgram();
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, num_vertex);

    glfwSwapBuffers(window_);
}

void GLRender::DrawPoint(const GLuint vao, int num_vertex)
{
    ProcessInput(window_);
    // render something
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader_program_.UseProgram();
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, num_vertex);

    glfwSwapBuffers(window_);
    glfwPollEvents();
}

void GLRender::DrawTest(const GLuint vao, int num_vertex)
{
    ProcessInput(window_);
    // render something
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    shader_program_.UseProgram();
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, num_vertex);
    glDrawArrays(GL_POINTS, 0, num_vertex);

    glfwSwapBuffers(window_);
    glfwPollEvents();
}

void GLRender::DrawTriangleViewControl(const GLuint vao, int num_vertex)
{
    // per-frame time logic
    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    // input
    ProcessInput(window_);

    // render something
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader_program_.UseProgram();

    glm::mat4 projection = glm::perspective(
        glm::radians(camera_.Zoom),
        (float)SCR_WIDTH / (float)SCR_HEIGHT,
        0.1f,
        1000.0f);
    shader_program_.SetMat4("projection", projection);

    glm::mat4 view = camera_.GetViewMatrix();
    shader_program_.SetMat4("view", view);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, num_vertex);

    glfwSwapBuffers(window_);
    glfwPollEvents();
}
