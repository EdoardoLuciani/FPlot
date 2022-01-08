mod renderer;

use std::time::{Duration, Instant};
use renderer::vk::graph_vk::GraphVk;
use crate::renderer::window_manager::WindowManager;

use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::platform::run_return::EventLoopExtRunReturn;

use nalgebra::*;

fn main() {
    std::env::set_var("WINIT_UNIX_BACKEND", "x11");
    let mut window = WindowManager::new((800u32, 800u32), None);
    let mut gvk = GraphVk::new((800u32, 800u32), window.get_window_handle());
    let mut center = Vector3::new(0.0f32, 0.0f32, 0.0f32);
    gvk.set_position(&center);
    gvk.fill_graph_buffer(center.x - 1.0f32, center.x + 1.0f32,|x| x*x);
    gvk.prepare();
    window.event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {event, ..} => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                },
                WindowEvent::KeyboardInput {
                    input:
                    KeyboardInput {
                        virtual_keycode: Some(vkc),
                        state: ElementState::Pressed,
                        ..
                    },
                    ..
                } => match vkc {
                    VirtualKeyCode::Up => { center.y += 0.01f32; window.window.request_redraw()},
                    VirtualKeyCode::Down => { center.y -= 0.01f32; window.window.request_redraw()},
                    VirtualKeyCode::Left => { center.x += 0.01f32; window.window.request_redraw()},
                    VirtualKeyCode::Right => { center.x -= 0.01f32; window.window.request_redraw()},
                    _ => ()
                },
                _ => (),
            },
            Event::RedrawRequested(_) => {
                gvk.fill_graph_buffer(-center.x - 1.0f32, -center.x + 1.0f32,|x| x*x);
                gvk.set_position(&center);
                gvk.present_loop(&window.window);
            },
            _ => (),
        }
    });
}
