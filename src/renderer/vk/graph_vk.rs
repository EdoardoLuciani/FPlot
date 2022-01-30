use super::base_vk::*;
use ash::{extensions::*, vk};
use gpu_allocator::MemoryLocation;
use nalgebra::*;
use raw_window_handle::RawWindowHandle;
use std::ffi::CStr;
use std::mem::size_of;

struct FrameData {
    after_exec_fence: vk::Fence,
    main_command: CommandRecordInfo,
}

pub struct GraphVk {
    bvk: BaseVk,
    sync2: khr::Synchronization2,
    host_curve_buffer: BufferAllocation,
    device_curve_buffer: BufferAllocation,
    transform_uniform_buffer: BufferAllocation,
    frames_data: Vec<FrameData>,
    renderpass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_info: DescriptorInfo,
    framebuffer: vk::Framebuffer,
    semaphores: Vec<vk::Semaphore>,
    frames_count: u64,
}

impl GraphVk {
    pub fn new(window_size: (u32, u32), window_handle: RawWindowHandle) -> Self {
        let mut imageless_fb =
            vk::PhysicalDeviceImagelessFramebufferFeatures::builder().imageless_framebuffer(true);
        let mut sync2 =
            vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
        let mut desired_features = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut sync2)
            .push_next(&mut imageless_fb);
        desired_features.features.fill_mode_non_solid = vk::TRUE;
        let mut base_vk = BaseVk::new(
            "FPlot",
            &[],
            &[
                "VK_KHR_synchronization2",
                "VK_KHR_imageless_framebuffer",
                "VK_KHR_image_format_list",
            ],
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
        let sync2 = khr::Synchronization2::new(&base_vk.instance, &base_vk.device);
        let buffers = Self::create_curve_vertex_buffers(&mut base_vk);

        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(std::mem::size_of::<nalgebra::Matrix4<f32>>() as u64)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER);
        let transform_uniform_buffer =
            base_vk.allocate_buffer(&buffer_create_info, MemoryLocation::CpuToGpu);

        let renderpass = Self::create_renderpass(&base_vk);
        let (descriptor_set_layout, descriptor_pool_size) =
            Self::create_descriptor_set_layout(&base_vk);
        let pipeline_data = Self::create_graph_pipeline(
            &base_vk,
            std::path::Path::new("assets/shaders-spirv"),
            renderpass,
            descriptor_set_layout,
        );
        let descriptor_info = base_vk.create_descriptor_pool_and_sets(
            std::slice::from_ref(&descriptor_pool_size),
            std::slice::from_ref(&descriptor_set_layout),
        );
        let framebuffer = Self::create_framebuffer(&base_vk, renderpass);
        let semaphores = base_vk.create_semaphores(2);

        let fence_create_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let frames_data = (0..3)
            .map(|_| FrameData {
                after_exec_fence: unsafe {
                    base_vk
                        .device
                        .create_fence(&fence_create_info, None)
                        .unwrap()
                },
                main_command: base_vk.create_cmd_pool_and_buffers(
                    vk::CommandPoolCreateFlags::empty(),
                    vk::CommandBufferLevel::PRIMARY,
                    base_vk.swapchain_image_views.as_ref().unwrap().len() as u32,
                ),
            })
            .collect();
        GraphVk {
            bvk: base_vk,
            sync2,
            host_curve_buffer: buffers[0].clone(),
            device_curve_buffer: buffers[1].clone(),
            transform_uniform_buffer,
            frames_data,
            renderpass,
            descriptor_set_layout,
            pipeline_layout: pipeline_data.0,
            pipeline: pipeline_data.1,
            descriptor_info,
            framebuffer,
            semaphores,
            frames_count: 0,
        }
    }

    fn get_required_vertex_buffer_size(bvk: &BaseVk) -> usize {
        (bvk.swapchain_create_info
            .as_ref()
            .unwrap()
            .image_extent
            .width as usize
            + 4)
            * (2 * std::mem::size_of::<f32>())
    }

    fn create_curve_vertex_buffers(bvk: &mut BaseVk) -> [BufferAllocation; 2] {
        // The size required for the buffers is calculated as the size of the points and the axes
        let mut buffer_create_info = vk::BufferCreateInfo::builder()
            .size(Self::get_required_vertex_buffer_size(bvk) as u64)
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

    fn recreate_curve_vertex_buffers(&mut self) {
        if Self::get_required_vertex_buffer_size(&self.bvk)
            > self.host_curve_buffer.allocation.size() as usize
        {
            self.bvk.destroy_buffer(&self.host_curve_buffer);
            self.bvk.destroy_buffer(&self.device_curve_buffer);

            let v = Self::create_curve_vertex_buffers(&mut self.bvk);
            self.host_curve_buffer = v[0].clone();
            self.device_curve_buffer = v[1].clone();
        }
    }

    pub fn fill_graph_buffer(&mut self, x_start: f32, x_end: f32, fun: fn(f32) -> f32) {
        let step = ((x_end - x_start).abs())
            / self.bvk.swapchain_create_info.unwrap().image_extent.width as f32;
        let mut x = x_start;
        let data_slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.host_curve_buffer
                    .allocation
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr() as *mut [f32; 2],
                self.host_curve_buffer.allocation.size() as usize / std::mem::size_of::<[f32; 2]>(),
            )
        };

        // points of the x axis
        data_slice[0][0] = x_start;
        data_slice[0][1] = 0.0f32;
        data_slice[1][0] = x_end;
        data_slice[1][1] = 0.0f32;

        for data in data_slice.iter_mut().skip(4) {
            data[0] = x;
            data[1] = fun(x);
            x += step;
        }
    }

    pub fn set_transform(&mut self, position: &Vector3<f32>, scale: f32) {
        let data_slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.host_curve_buffer
                    .allocation
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr() as *mut [f32; 2],
                self.host_curve_buffer.allocation.size() as usize / std::mem::size_of::<[f32; 2]>(),
            )
        };

        // points of the y axis
        data_slice[2][0] = 0.0f32;
        data_slice[2][1] = position.y + 1.0f32 / scale;
        data_slice[3][0] = 0.0f32;
        data_slice[3][1] = position.y - 1.0f32 / scale;

        let translation = Matrix4::<f32>::new_translation(position);
        let scaling = Matrix4::<f32>::new_scaling(scale);
        let transform = scaling * translation;
        let dst_ptr = std::ptr::slice_from_raw_parts_mut(
            self.transform_uniform_buffer
                .allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr() as *mut f32,
            self.transform_uniform_buffer.allocation.size() as usize / std::mem::size_of::<f32>(),
        );
        unsafe {
            (*dst_ptr).copy_from_slice(transform.data.as_slice());
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
        unsafe {
            bvk.device
                .create_render_pass(&renderpass_create_info, None)
                .unwrap()
        }
    }

    fn create_descriptor_set_layout(
        bvk: &BaseVk,
    ) -> (vk::DescriptorSetLayout, vk::DescriptorPoolSize) {
        let descriptor_bindings: [vk::DescriptorSetLayoutBinding; 1] =
            [vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .build()];
        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_bindings);
        let dsl = unsafe {
            bvk.device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .unwrap()
        };
        (
            dsl,
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
        )
    }

    fn create_graph_pipeline(
        bvk: &BaseVk,
        shader_dir: &std::path::Path,
        renderpass: vk::RenderPass,
        descriptor_set_layout: vk::DescriptorSetLayout,
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
                .topology(vk::PrimitiveTopology::LINE_STRIP)
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
                .polygon_mode(vk::PolygonMode::LINE)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .depth_bias_enable(false)
                .line_width(1.0f32);

        let pipeline_multisample_state_create_info =
            vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build();
        let pipeline_color_blend_state_create_info =
            vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .attachments(std::slice::from_ref(&color_blend_attachment_state));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let pipeline_dynamic_state_create_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<Vector4<f32>>() as u32);
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(std::slice::from_ref(&descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
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

    fn create_framebuffer(bvk: &BaseVk, renderpass: vk::RenderPass) -> vk::Framebuffer {
        let framebuffer_attachments_image_info = vk::FramebufferAttachmentImageInfo::builder()
            .usage(bvk.swapchain_create_info.unwrap().image_usage)
            .width(bvk.swapchain_create_info.unwrap().image_extent.width)
            .height(bvk.swapchain_create_info.unwrap().image_extent.height)
            .layer_count(1)
            .view_formats(std::slice::from_ref(
                &bvk.swapchain_create_info.as_ref().unwrap().image_format,
            ));
        let mut framebuffer_attachments_create_info =
            vk::FramebufferAttachmentsCreateInfoKHR::builder()
                .attachment_image_infos(std::slice::from_ref(&framebuffer_attachments_image_info));
        let mut framebuffer_create_info = vk::FramebufferCreateInfo::builder()
            .push_next(&mut framebuffer_attachments_create_info)
            .flags(vk::FramebufferCreateFlags::IMAGELESS_KHR)
            .render_pass(renderpass)
            .width(bvk.swapchain_create_info.unwrap().image_extent.width)
            .height(bvk.swapchain_create_info.unwrap().image_extent.height)
            .layers(1);
        // todo: maybe remove width and height field from framebuffer_create_info, as the imageless framebuffer is used
        framebuffer_create_info.attachment_count = 1;
        unsafe {
            bvk.device
                .create_framebuffer(&framebuffer_create_info, None)
                .unwrap()
        }
    }

    pub fn prepare(&mut self) {
        self.write_descriptor_sets();
        self.frames_data
            .iter()
            .for_each(|e| self.record_static_command_buffers(&e.main_command));
    }

    fn write_descriptor_sets(&self) {
        let descriptor_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(self.transform_uniform_buffer.buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let write_descriptor_sets = vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_info.buffers[0])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&descriptor_buffer_info));
        unsafe {
            self.bvk
                .device
                .update_descriptor_sets(std::slice::from_ref(&write_descriptor_sets), &[]);
        }
    }

    fn record_static_command_buffers(&self, cmri: &CommandRecordInfo) {
        unsafe {
            self.bvk
                .device
                .reset_command_pool(cmri.pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
        for (i, cmd_buf) in cmri.buffers.iter().enumerate() {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
            unsafe {
                self.bvk
                    .device
                    .begin_command_buffer(*cmd_buf, &command_buffer_begin_info)
                    .unwrap();
                let region = vk::BufferCopy::builder()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(Self::get_required_vertex_buffer_size(&self.bvk) as u64);
                self.bvk.device.cmd_copy_buffer(
                    *cmd_buf,
                    self.host_curve_buffer.buffer,
                    self.device_curve_buffer.buffer,
                    std::slice::from_ref(&region),
                );

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
                let dependency_info = vk::DependencyInfoKHR::builder()
                    .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barrier));
                self.sync2.cmd_pipeline_barrier2(*cmd_buf, &dependency_info);

                let mut renderpass_attachment_begin_info =
                    vk::RenderPassAttachmentBeginInfoKHR::builder().attachments(
                        std::slice::from_ref(&self.bvk.swapchain_image_views.as_ref().unwrap()[i]),
                    );
                let mut clear_values = vk::ClearValue::default();
                clear_values.color.float32 = [0.0f32, 0.0f32, 0.0f32, 0.0f32];
                let renderpass_begin_info = vk::RenderPassBeginInfo::builder()
                    .push_next(&mut renderpass_attachment_begin_info)
                    .render_pass(self.renderpass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.bvk.swapchain_create_info.unwrap().image_extent,
                    })
                    .clear_values(std::slice::from_ref(&clear_values));
                self.bvk.device.cmd_begin_render_pass(
                    *cmd_buf,
                    &renderpass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                self.bvk.device.cmd_bind_pipeline(
                    *cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline,
                );
                let viewport = vk::Viewport::builder()
                    .x(0.0f32)
                    .y(0.0f32)
                    .width(self.bvk.swapchain_create_info.unwrap().image_extent.width as f32)
                    .height(self.bvk.swapchain_create_info.unwrap().image_extent.height as f32)
                    .min_depth(0.0f32)
                    .max_depth(1.0f32);
                self.bvk
                    .device
                    .cmd_set_viewport(*cmd_buf, 0, std::slice::from_ref(&viewport));
                let scissor = vk::Rect2D::builder()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(self.bvk.swapchain_create_info.unwrap().image_extent);
                self.bvk
                    .device
                    .cmd_set_scissor(*cmd_buf, 0, std::slice::from_ref(&scissor));
                self.bvk.device.cmd_bind_descriptor_sets(
                    *cmd_buf,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    std::slice::from_ref(&self.descriptor_info.buffers[0]),
                    &[],
                );
                self.bvk.device.cmd_bind_vertex_buffers(
                    *cmd_buf,
                    0,
                    std::slice::from_ref(&self.device_curve_buffer.buffer),
                    std::slice::from_ref(&0),
                );
                // Drawing of the axes
                let axes_color = Vector4::<f32>::new(1.0, 0.0, 0.0, 0.0);
                self.bvk.device.cmd_push_constants(
                    *cmd_buf,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_raw_parts(axes_color.as_ptr() as *const u8, 16),
                );
                self.bvk.device.cmd_draw(*cmd_buf, 2, 1, 0, 0);
                self.bvk.device.cmd_draw(*cmd_buf, 2, 1, 2, 0);
                // Drawing of the function
                let function_color = Vector4::<f32>::new(1.0, 1.0, 1.0, 0.0);
                self.bvk.device.cmd_push_constants(
                    *cmd_buf,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    std::slice::from_raw_parts(function_color.as_ptr() as *const u8, 16),
                );
                self.bvk.device.cmd_draw(
                    *cmd_buf,
                    self.bvk.swapchain_create_info.unwrap().image_extent.width,
                    1,
                    4,
                    0,
                );
                self.bvk.device.cmd_end_render_pass(*cmd_buf);
                self.bvk.device.end_command_buffer(*cmd_buf).unwrap();
            }
        }
    }

    pub fn present_loop(&mut self, window: &winit::window::Window) {
        let current_frame_data =
            &self.frames_data[self.frames_count as usize % self.frames_data.len()];
        unsafe {
            let res = self.bvk.swapchain_fn.as_ref().unwrap().acquire_next_image(
                self.bvk.swapchain,
                u64::MAX,
                self.semaphores[0],
                vk::Fence::null(),
            );
            // if the swapchain is suboptimal
            if res.is_err() || res.unwrap().1 {
                self.bvk
                    .device
                    .device_wait_idle()
                    .expect("Device wait failed");
                self.bvk.recreate_swapchain(
                    self.bvk.swapchain_create_info.unwrap().present_mode,
                    vk::Extent2D {
                        width: window.inner_size().width,
                        height: window.inner_size().height,
                    },
                    self.bvk.swapchain_create_info.unwrap().image_usage,
                    vk::SurfaceFormatKHR {
                        format: self.bvk.swapchain_create_info.unwrap().image_format,
                        color_space: self.bvk.swapchain_create_info.unwrap().image_color_space,
                    },
                );
                self.bvk.device.destroy_framebuffer(self.framebuffer, None);
                self.framebuffer = Self::create_framebuffer(&self.bvk, self.renderpass);
                self.recreate_curve_vertex_buffers();
                self.prepare();
                return;
            }
            let res = res.unwrap();
            self.bvk
                .device
                .wait_for_fences(
                    std::slice::from_ref(&current_frame_data.after_exec_fence),
                    false,
                    u64::MAX,
                )
                .expect("Fence wait failed");
            self.bvk
                .device
                .reset_fences(std::slice::from_ref(&current_frame_data.after_exec_fence))
                .expect("Fence reset failed");

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
            self.sync2
                .queue_submit2(
                    self.bvk.queues[0],
                    std::slice::from_ref(&submit_info),
                    current_frame_data.after_exec_fence,
                )
                .expect("Error submitting queue");

            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(std::slice::from_ref(&self.semaphores[1]))
                .swapchains(std::slice::from_ref(&self.bvk.swapchain))
                .image_indices(std::slice::from_ref(&res.0));
            self.bvk
                .swapchain_fn
                .as_ref()
                .unwrap()
                .queue_present(self.bvk.queues[0], &present_info)
                .expect("Queue present failed");

            self.frames_count += 1;
        }
    }
}

impl Drop for GraphVk {
    fn drop(&mut self) {
        unsafe {
            self.bvk.device.device_wait_idle().unwrap();
            self.bvk
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        };
        self.bvk
            .destroy_descriptor_pool_and_sets(&self.descriptor_info);
        self.bvk.destroy_semaphores(&self.semaphores);
        self.frames_data.iter().for_each(|frame_data| {
            unsafe {
                self.bvk
                    .device
                    .destroy_fence(frame_data.after_exec_fence, None)
            };
            self.bvk
                .destroy_cmd_pool_and_buffers(&frame_data.main_command);
        });

        self.bvk.destroy_buffer(&self.host_curve_buffer);
        self.bvk.destroy_buffer(&self.device_curve_buffer);
        self.bvk.destroy_buffer(&self.transform_uniform_buffer);

        unsafe { self.bvk.device.destroy_framebuffer(self.framebuffer, None) };
        unsafe {
            self.bvk
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.bvk.device.destroy_pipeline(self.pipeline, None);
            self.bvk.device.destroy_render_pass(self.renderpass, None);
        }
    }
}
