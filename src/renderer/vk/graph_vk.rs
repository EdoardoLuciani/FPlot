use std::mem::size_of;
use ash::{extensions::*, vk};
use std::slice;
use super::base_vk::*;
use super::super::window_manager::WindowManager;
use gpu_allocator::vulkan as vkalloc;

pub struct GraphVk {
    bvk: BaseVk,
    window: WindowManager,
    host_curve_buffer: BufferAllocation,
    device_curve_buffer: BufferAllocation
}

impl GraphVk {
    pub fn new(window_size: (u32, u32)) -> Self {
        let mut desired_features = vk::PhysicalDeviceFeatures2::default();
        let mut window = WindowManager::new(window_size, None);
        let mut base_vk = BaseVk::new(
            "FPlot",
            &[],
            &[],
            &desired_features,
            slice::from_ref(&(vk::QueueFlags::GRAPHICS, 1.0f32)),
            Some(window.get_window_handle()),
        );
        base_vk.recreate_swapchain(
            vk::PresentModeKHR::MAILBOX,
            vk::Extent2D {
                width: window_size.0,
                height: window_size.1,
            },
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        );
        let buffers = Self::create_curve_vertex_buffers(&mut base_vk, (window_size.0 as usize * 2 * size_of::<f32>()) as u64);
        GraphVk{bvk: base_vk, window, host_curve_buffer: buffers[0].clone(), device_curve_buffer: buffers[1].clone()}
    }

    fn create_curve_vertex_buffers(bvk: &mut BaseVk, size: u64) -> [BufferAllocation; 2] {
        let mut buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let host_buffer = bvk.allocate_buffer(&buffer_create_info, gpu_allocator::MemoryLocation::GpuToCpu);

        buffer_create_info.usage = vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let device_buffer = bvk.allocate_buffer(&buffer_create_info, gpu_allocator::MemoryLocation::GpuOnly);
        [host_buffer, device_buffer]
    }

    fn recreate_curve_vertex_buffers(&mut self, size: u64) {
        if size > self.host_curve_buffer.allocation.size() {
            self.bvk.destroy_buffer(&self.host_curve_buffer);
            self.bvk.destroy_buffer(&self.device_curve_buffer);

            let v = Self::create_curve_vertex_buffers(&mut self.bvk, size);
            self.host_curve_buffer = v[0].clone();
            self.device_curve_buffer = v[1].clone();
        }
    }

    fn create_graph_pipeline(bvk: &mut BaseVk) {
        bvk.device.create_graphics_pipelines()
    }
}

impl Drop for GraphVk {
    fn drop(&mut self) {
        self.bvk.destroy_buffer(&self.host_curve_buffer);
        self.bvk.destroy_buffer(&self.device_curve_buffer);
    }
}