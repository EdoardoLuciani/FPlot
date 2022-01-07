#version 460

layout (location = 0) in vec2 position;

layout (set = 0, binding = 0) uniform uniform_buffer {
    mat4 transform;
};

void main() {
    gl_Position = transform * vec4(position.x, -position.y, 0.0f, 1.0f);
}