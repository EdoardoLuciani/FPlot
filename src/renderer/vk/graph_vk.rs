use std::borrow::{Borrow, BorrowMut};
use super::base_vk::*;
use ash::{extensions::*, vk};
use std::ffi::CStr;
use raw_window_handle::RawWindowHandle;
use rand::distributions::{Distribution, Uniform};

struct FrameData {
    after_exec_fence: vk::Fence,
    main_command: CommandRecordInfo,
}

pub struct GraphVk {
    bvk: BaseVk,
    sync2: khr::Synchronization2,
    host_curve_buffer: BufferAllocation,
    device_curve_buffer: BufferAllocation,
    frames_data: Vec<FrameData>,
    renderpass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    framebuffers: Vec<vk::Framebuffer>,
    semaphores: Vec<vk::Semaphore>,
    frames_count: u64,
}

impl GraphVk {
    pub fn new(window_size: (u32, u32), window_handle: RawWindowHandle) -> Self {
        let mut sync2 = vk::PhysicalDeviceSynchronization2FeaturesKHR::builder()
            .synchronization2(true);
        let mut desired_features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut sync2);
        desired_features.features.fill_mode_non_solid = vk::TRUE;
        let mut base_vk = BaseVk::new(
            "FPlot",
            &[],
            &["VK_KHR_synchronization2"],
            &desired_features,
            std::slice::from_ref(&(vk::QueueFlags::GRAPHICS, 1.0f32)),
            Some(window_handle),
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
        let sync2 =  khr::Synchronization2::new(&base_vk.instance, &base_vk.device);
        let buffers = Self::create_curve_vertex_buffers(
            &mut base_vk,
            (window_size.0 as usize * 2 * std::mem::size_of::<f32>()) as u64,
        );
        let renderpass = Self::create_renderpass(&mut base_vk);
        let pipeline_data = Self::create_graph_pipeline(
            &mut base_vk,
            std::path::Path::new("assets/shaders-spirv"),
            renderpass,
        );
        let framebuffers = Self::create_framebuffers(&mut base_vk, renderpass);
        let semaphores = base_vk.create_semaphores(2);

        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);
        let frames_data= (0..3).map(|_|
            FrameData {
                after_exec_fence: unsafe {base_vk.device.create_fence(&fence_create_info, None).unwrap()},
                main_command: base_vk.create_cmd_pool_and_buffers(
                    vk::CommandPoolCreateFlags::empty(),
                    vk::CommandBufferLevel::PRIMARY,
                    base_vk.swapchain_image_views.as_ref().unwrap().len() as u32,
                )
            }
        ).collect();
        GraphVk {
            bvk: base_vk,
            sync2,
            host_curve_buffer: buffers[0].clone(),
            device_curve_buffer: buffers[1].clone(),
            frames_data,
            renderpass,
            pipeline_layout: pipeline_data.0,
            pipeline: pipeline_data.1,
            framebuffers,
            semaphores,
            frames_count: 0,
        }
    }

    fn create_curve_vertex_buffers(bvk: &mut BaseVk, size: u64) -> [BufferAllocation; 2] {
        let mut buffer_create_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .build();
        let host_buffer =
            bvk.allocate_buffer(&buffer_create_info, gpu_allocator::MemoryLocation::GpuToCpu);

        buffer_create_info.usage =
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let device_buffer =
            bvk.allocate_buffer(&buffer_create_info, gpu_allocator::MemoryLocation::GpuOnly);
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

    fn create_renderpass(bvk: &BaseVk) -> vk::RenderPass {
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
            bvk.device
                .create_render_pass(&renderpass_create_info, None)
                .unwrap()
        };
        renderpass
    }

    fn create_graph_pipeline(
        bvk: &BaseVk,
        shader_dir: &std::path::Path,
        renderpass: vk::RenderPass,
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        // Creating the shader modules
        let vertex_shader = super::get_binary_shader_data(shader_dir.join("vertex.vert.spirv"));
        let vertex_shader_module = unsafe {
            bvk.device
                .create_shader_module(&vertex_shader.2, None)
                .unwrap()
        };
        let fragment_shader = super::get_binary_shader_data(shader_dir.join("fragment.frag.spirv"));
        let fragment_shader_module = unsafe {
            bvk.device
                .create_shader_module(&fragment_shader.2, None)
                .unwrap()
        };
        let shader_entry_point_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
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
            .stride(2 * std::mem::size_of::<f32>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
        let vertex_input_attribute: [vk::VertexInputAttributeDescription; 1] =
            [vk::VertexInputAttributeDescription::builder()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0)
                .build()];
        let pipeline_vertex_input_state_create_info =
            vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(std::slice::from_ref(&vertex_input_binding))
                .vertex_attribute_descriptions(&vertex_input_attribute);

        let pipeline_input_assembly_create_info =
            vk::PipelineInputAssemblyStateCreateInfo::builder()
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
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D {
                width: 64,
                height: 64,
            })
            .build();
        let pipeline_viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let pipeline_rasterization_state_create_info =
            vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0f32);

        let pipeline_multisample_state_create_info =
            vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .build();
        let pipeline_color_blend_state_create_info =
            vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .attachments(std::slice::from_ref(&color_blend_attachment_state));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let pipeline_dynamic_state_create_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            bvk.device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
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
            bvk.device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&graphics_pipeline_create_info),
                    None,
                )
                .unwrap()
        };

        unsafe {
            bvk.device.destroy_shader_module(vertex_shader_module, None);
            bvk.device
                .destroy_shader_module(fragment_shader_module, None);
        }
        (pipeline_layout, pipeline[0])
    }

    fn create_framebuffers(bvk: &BaseVk, renderpass: vk::RenderPass) -> Vec<vk::Framebuffer> {
        let mut out_vec = Vec::new();
        for swapchain_image_view in bvk.swapchain_image_views.as_ref().unwrap().iter() {
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(renderpass)
                .attachments(std::slice::from_ref(swapchain_image_view))
                .width(bvk.swapchain_create_info.unwrap().image_extent.width)
                .height(bvk.swapchain_create_info.unwrap().image_extent.height)
                .layers(1);
            out_vec.push(unsafe {
                bvk.device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .unwrap()
            });
        }
        out_vec
    }

    fn record_static_command_buffers(&self, cmri: &CommandRecordInfo) {
        unsafe {
            self.bvk.device.reset_command_pool(cmri.pool, vk::CommandPoolResetFlags::empty());
        }
        for (i, cmd_buf) in cmri.buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
            unsafe {
                self.bvk.device.begin_command_buffer(*cmd_buf, &command_buffer_begin_info);
                let region = vk::BufferCopy::builder()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(self.host_curve_buffer.allocation.size());
                self.bvk.device.cmd_copy_buffer(*cmd_buf, self.host_curve_buffer.buffer, self.device_curve_buffer.buffer, std::slice::from_ref(&region));

                let buffer_memory_barrier = vk::BufferMemoryBarrier2KHR::builder()
                    .src_stage_mask(vk::PipelineStageFlags2KHR::COPY)
                    .src_access_mask(vk::AccessFlags2KHR::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2KHR::VERTEX_ATTRIBUTE_INPUT)
                    .dst_access_mask(vk::AccessFlags2KHR::VERTEX_ATTRIBUTE_READ)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(self.device_curve_buffer.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                let dependancy_info = vk::DependencyInfoKHR::builder()
                    .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barrier));
                self.sync2.cmd_pipeline_barrier2(*cmd_buf, &dependancy_info);

                let mut clear_values = vk::ClearValue::default();
                clear_values.color.float32 = [0.0f32, 0.0f32, 0.0f32, 0.0f32];
                let renderpass_begin_info = vk::RenderPassBeginInfo::builder()
                    .render_pass(self.renderpass)
                    .framebuffer(self.framebuffers[i])
                    .render_area(vk::Rect2D{offset: vk::Offset2D{x: 0, y: 0}, extent: self.bvk.swapchain_create_info.unwrap().image_extent})
                    .clear_values(std::slice::from_ref(&clear_values));
                self.bvk.device.cmd_begin_render_pass(*cmd_buf, &renderpass_begin_info, vk::SubpassContents::INLINE);

                self.bvk.device.cmd_bind_pipeline(*cmd_buf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                let viewport = vk::Viewport::builder()
                    .x(0.0f32)
                    .y(0.0f32)
                    .width(self.bvk.swapchain_create_info.unwrap().image_extent.width as f32)
                    .height(self.bvk.swapchain_create_info.unwrap().image_extent.height as f32)
                    .min_depth(0.0f32)
                    .max_depth(1.0f32);
                self.bvk.device.cmd_set_viewport(*cmd_buf, 0, std::slice::from_ref(&viewport));
                let scissor = vk::Rect2D::builder()
                    .offset(vk::Offset2D{x:0, y:0})
                    .extent(self.bvk.swapchain_create_info.unwrap().image_extent);
                self.bvk.device.cmd_set_scissor(*cmd_buf, 0, std::slice::from_ref(&scissor));
                self.bvk.device.cmd_bind_vertex_buffers(*cmd_buf, 0, std::slice::from_ref(&self.device_curve_buffer.buffer), std::slice::from_ref(&0));
                //self.bvk.device.cmd_draw(*cmd_buf, self.bvk.swapchain_create_info.unwrap().image_extent.width, 1, 0, 0);
                self.bvk.device.cmd_draw(*cmd_buf, 3, 1, 0, 0);
                self.bvk.device.cmd_end_render_pass(*cmd_buf);
                self.bvk.device.end_command_buffer(*cmd_buf);
            }

        }
    }

    pub fn fill_graph_buffer(&mut self, fun : fn(f32) -> f32) {
        let step = 2.0f32/self.bvk.swapchain_create_info.unwrap().image_extent.width as f32;
        let mut x = -1.0f32;
        let data_slice = unsafe { std::slice::from_raw_parts_mut(self.host_curve_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut [f32; 2], self.bvk.swapchain_create_info.unwrap().image_extent.width as usize) };
        let mut rng = rand::thread_rng();
        let uniform_dist = Uniform::from(0.0f32..1.0f32);
        for i in 0..self.bvk.swapchain_create_info.unwrap().image_extent.width as usize {
            /*
            data_slice[i][0] = 0.0f32;
            data_slice[i][1] = fun(x);
            */
            data_slice[i][0] = uniform_dist.sample(&mut rng);
            data_slice[i][1] = uniform_dist.sample(&mut rng);
            x += step;
        }
    }

    pub fn prepare(&mut self) {
        for i in 0..self.frames_data.len() {
            self.record_static_command_buffers(&self.frames_data[i].main_command);
        }
        //self.frames_data.iter_mut().for_each(|e| self.record_static_command_buffers(&mut e.main_command));
    }

    pub fn present_loop(&mut self, window: &winit::window::Window) {
        let current_frame_data = &self.frames_data[self.frames_count as usize % self.frames_data.len()];
        unsafe {
            let res = self.bvk.swapchain_fn.as_ref().unwrap().acquire_next_image(self.bvk.swapchain, u64::MAX, self.semaphores[0], vk::Fence::null()).unwrap();
            // if the swapchain is suboptimal
            if res.1 {
                unsafe {
                    self.bvk.device.device_wait_idle();
                }
                self.bvk.recreate_swapchain(
                    vk::PresentModeKHR::MAILBOX,
                    vk::Extent2D {
                        width: window.inner_size().width,
                        height: window.inner_size().height
                    },
                    vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8_UNORM,
                        color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
                    },
                );
                self.framebuffers = Self::create_framebuffers(&self.bvk, self.renderpass);
                self.prepare();
                return
            }
            self.bvk.device.wait_for_fences(std::slice::from_ref(&current_frame_data.after_exec_fence), false, u64::MAX);
            self.bvk.device.reset_fences(std::slice::from_ref(&current_frame_data.after_exec_fence));

            let wait_semaphore_submit_info = vk::SemaphoreSubmitInfoKHR::builder()
                .semaphore(self.semaphores[0])
                .stage_mask(vk::PipelineStageFlags2KHR::COLOR_ATTACHMENT_OUTPUT)
                .device_index(0);
            let command_submit_info = vk::CommandBufferSubmitInfoKHR::builder()
                .command_buffer(current_frame_data.main_command.buffers[res.0 as usize])
                .device_mask(0);
            let signal_semaphore_submit_info = vk::SemaphoreSubmitInfoKHR::builder()
                .semaphore(self.semaphores[1])
                .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
                .device_index(0);
            let submit_info = vk::SubmitInfo2KHR::builder()
                .wait_semaphore_infos(std::slice::from_ref(&wait_semaphore_submit_info))
                .command_buffer_infos(std::slice::from_ref(&command_submit_info))
                .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_submit_info))
                .build();
            self.sync2.queue_submit2(self.bvk.queues[0], std::slice::from_ref(&submit_info), current_frame_data.after_exec_fence);

            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(std::slice::from_ref(&self.semaphores[1]))
                .swapchains(std::slice::from_ref(&self.bvk.swapchain))
                .image_indices(std::slice::from_ref(&res.0));
            self.bvk.swapchain_fn.as_ref().unwrap().queue_present(self.bvk.queues[0], &present_info);
            self.frames_count += 1;
        }
    }
}

impl Drop for GraphVk {
    fn drop(&mut self) {
        unsafe { self.bvk.device.device_wait_idle() };
        self.bvk.destroy_semaphores(&self.semaphores);
        self.frames_data.iter().for_each(|frame_data| {
            unsafe { self.bvk.device.destroy_fence(frame_data.after_exec_fence, None) };
            self.bvk.destroy_cmd_pool_and_buffers(&frame_data.main_command);
        });

        self.bvk.destroy_buffer(&self.host_curve_buffer);
        self.bvk.destroy_buffer(&self.device_curve_buffer);

        self.framebuffers.iter().for_each(|framebuffer| {
            unsafe {
                self.bvk.device.destroy_framebuffer(*framebuffer, None);
            }
        });
        unsafe {
            self.bvk
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.bvk.device.destroy_pipeline(self.pipeline, None);
            self.bvk.device.destroy_render_pass(self.renderpass, None);
        }
    }
}
