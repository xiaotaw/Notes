#version 330 core
layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_color;

out vec3 fragment_color;

void main(){
    gl_Position = vec4(vertex_position.x, vertex_position.y, vertex_position.z, 1.0);
    gl_PointSize = 3.0;
    fragment_color = vertex_color;
}