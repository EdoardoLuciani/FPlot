#version 460

layout (location = 0) out vec4 frag_color;

layout(push_constant) uniform constants {
    vec4 line_color;
} pc;

layout (location = 0) in FS_IN {
    vec3 position;
} fs_in;

void main() {
    frag_color = vec4(pc.line_color.xyz, 1.0f);
}

