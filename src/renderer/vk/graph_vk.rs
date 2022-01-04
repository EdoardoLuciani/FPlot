use std::ffi::CStr;
use ash::{extensions::*, vk};
use super::base_vk::*;
use super::super::window_manager::WindowManager;

pub struct GraphVk {
    bvk: BaseVk,
    window: WindowManager,
    host_curve_buffer: BufferAllocation,
    device_curve_buffer: BufferAllocation,
    renderpass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>
}

impl GraphVk {
    pub fn new(window_size: (u32, u32)) -> Self {
        let mut desired_features = vk::PhysicalDeviceFeatures2::default();
        desired_features.features.fill_mode_non_solid = vk::TRUE;
        let window = WindowManager::new(window_size, None);
        let mut base_vk = BaseVk::new(
            "FPlot",
            &[],
            &[],
            &desired_features,
            std::slice::from_ref(&(vk::QueueFlags::GRAPHICS, 1.0f32)),
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
        let buffers = Self::create_curve_vertex_buffers(&mut base_vk, (window_size.0 as usize * 2 * std::mem::size_of::<f32>()) as u64);
        let renderpass = Self::create_renderpass(&mut base_vk);
        let pipeline_data = Self::create_graph_pipeline(&mut base_vk, std::path::Path::new("assets/shaders-spirv"), renderpass);
        let framebuffers = Self::create_framebuffers(&mut base_vk, renderpass);
        GraphVk{bvk: base_vk, window, host_curve_buffer: buffers[0].clone(), device_curve_buffer: buffers[1].clone(), renderpass, pipeline_layout: pipeline_data.0, pipeline: pipeline_data.1, framebuffers}
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

    fn create_renderpass(bvk: &mut BaseVk) -> vk::RenderPass {
        let attachment_descriptions = vk::AttachmentDescription::builder()
            .format(bvk.swapchain_create_info.unwrap().image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let attachment_references = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let subpass_description = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&attachment_references));

        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(std::slice::from_ref(&attachment_descriptions))
            .subpasses(std::slice::from_ref(&subpass_description));
        let renderpass = unsafe {
            bvk.device.create_render_pass(&renderpass_create_info, None).unwrap()
        };
        renderpass
    }

    fn create_graph_pipeline(bvk: &mut BaseVk, shader_dir: &std::path::Path, renderpass: vk::RenderPass) -> (vk::PipelineLayout, vk::Pipeline) {
        // Creating the shader modules
        let vertex_shader = super::get_binary_shader_data(shader_dir.join("vertex.vert.spirv"));
        let vertex_shader_module = unsafe {
            bvk.device.create_shader_module(&vertex_shader.2, None).unwrap()
        };
        let fragment_shader = super::get_binary_shader_data(shader_dir.join("fragment.frag.spirv"));
        let fragment_shader_module = unsafe {
            bvk.device.create_shader_module(&fragment_shader.2, None).unwrap()
        };
        let shader_entry_point_name = unsafe {CStr::from_bytes_with_nul_unchecked(b"main\0")};
        let pipeline_shader_stage_create_infos: [vk::PipelineShaderStageCreateInfo; 2] = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vertex_shader.1)
                .module(vertex_shader_module)
                .name(shader_entry_point_name)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(fragment_shader.1)
                .module(fragment_shader_module)
                .name(shader_entry_point_name)
                .build(),
        ];

        // Vertex state definition
        let vertex_input_binding = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(2*std::mem::size_of::<f32>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
        let vertex_input_attribute: [vk::VertexInputAttributeDescription ;1] = [
            vk::VertexInputAttributeDescription::builder()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0)
                .build()
        ];
        let pipeline_vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(std::slice::from_ref(&vertex_input_binding))
            .vertex_attribute_descriptions(&vertex_input_attribute);

        let pipeline_input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Dummy values for viewport and scissor since they will be set using dynamic states
        let viewport = vk::Viewport::builder()
            .x(0.0f32)
            .y(0.0f32)
            .width(64.0f32)
            .height(64.0f32)
            .min_depth(0.0f32)
            .max_depth(1.0f32)
            .build();
        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D{x: 0, y: 0})
            .extent(vk::Extent2D{width: 64, height: 64})
            .build();
        let pipeline_viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let pipeline_rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::POINT)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0f32);

        let pipeline_multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .build();
        let pipeline_color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment_state));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let pipeline_dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            bvk.device.create_pipeline_layout(&pipeline_layout_create_info, None).unwrap()
        };

        let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&pipeline_shader_stage_create_infos)
            .vertex_input_state(&pipeline_vertex_input_state_create_info)
            .input_assembly_state(&pipeline_input_assembly_create_info)
            .viewport_state(&pipeline_viewport_state_create_info)
            .rasterization_state(&pipeline_rasterization_state_create_info)
            .multisample_state(&pipeline_multisample_state_create_info)
            .color_blend_state(&pipeline_color_blend_state_create_info)
            .dynamic_state(&pipeline_dynamic_state_create_info)
            .layout(pipeline_layout)
            .render_pass(renderpass)
            .subpass(0);

        let pipeline = unsafe {
            bvk.device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&graphics_pipeline_create_info), None).unwrap()
        };

        unsafe {
            bvk.device.destroy_shader_module(vertex_shader_module, None);
            bvk.device.destroy_shader_module(fragment_shader_module, None);
        }
        (pipeline_layout, pipeline[0])
    }

    fn create_framebuffers(bvk: &mut BaseVk, renderpass: vk::RenderPass) -> Vec<vk::Framebuffer> {
        let mut out_vec = Vec::new();
        for swapchain_image_view in bvk.swapchain_image_views.as_ref().unwrap().iter() {
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass)
                .attachments(std::slice::from_ref(swapchain_image_view))
                .width(bvk.swapchain_create_info.unwrap().image_extent.width)
                .height(bvk.swapchain_create_info.unwrap().image_extent.height)
                .layers(1);
            out_vec.push(unsafe {
                bvk.device.create_framebuffer(&framebuffer_create_info, None).unwrap()
            });
        }
        out_vec
    }


}

impl Drop for GraphVk {
    fn drop(&mut self) {
        self.bvk.destroy_buffer(&self.host_curve_buffer);
        self.bvk.destroy_buffer(&self.device_curve_buffer);

        for framebuffer in self.framebuffers.iter() {
            unsafe {
                self.bvk.device.destroy_framebuffer(*framebuffer, None);
            }
        }
        unsafe {
            self.bvk.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.bvk.device.destroy_pipeline(self.pipeline, None);
            self.bvk.device.destroy_render_pass(self.renderpass, None);
        }
    }
}