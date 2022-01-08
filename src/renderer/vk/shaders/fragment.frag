#version 460

layout (location = 0) out vec4 frag_color;

layout (location = 0) in FS_IN {
    vec3 position;
} fs_in;

void main() {
    frag_color = vec4(1.0f, abs(tan(fs_in.position.y)), 0.0f, 1.0f);
}

