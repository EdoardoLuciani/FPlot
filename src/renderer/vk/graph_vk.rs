use ash::{extensions::*, vk};
use std::slice;
use super::base_vk::BaseVk;
use super::super::window_manager::WindowManager;

pub struct GraphVk {
    bvk: BaseVk,
    window: WindowManager
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
        GraphVk{bvk: base_vk, window}
    }
}