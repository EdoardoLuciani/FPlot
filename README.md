# FPlot
FPlot is a simple function visilizer. As of now, it is simply a toy project that I used to learn vulkan and window management for Rust and is not intended to be maintained. I decided to opensource it so it can be used as a reference.

![alt text](https://github.com/EdoardoLuciani/FPlot/img.png)

## Features
- Window panning, zoom
- Frame rendering occurs only on certain events to minimize resource consumption

## Interesting files
- build.rs automatically compiles the GLSL to SPIR-V
- BaseVk struct and impl provides a very comfortable interface for instance, device, extensions, queues and swapchain creation.

## How to build
The project can be built on any supported platform with cargo but it also requires the Vulkan SDK installation (https://vulkan.lunarg.com/). Important note is that the environment variables that the SDK provides (VULKAN_SDK, PATH, LD_LIBRARY_PATH, VK_LAYER_PATH) needs to be set as they are used by the various dependencies of FPlot.
