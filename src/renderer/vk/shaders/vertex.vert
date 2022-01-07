#version 460

layout (location = 0) in vec2 position;

void main() {
    vec2 tex_coord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(tex_coord * 2.0f + -1.0f, 0.0f, 1.0f);
}