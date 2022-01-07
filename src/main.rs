mod renderer;

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
    gvk.fill_graph_buffer(|x| x*x);
    let mut position = Vector3::new(0.0f32, 0.0f32, 0.0f32);
    gvk.prepare();
    window.event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
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
                    VirtualKeyCode::Up => {position.y += 0.01f32},
                    VirtualKeyCode::Down => {position.y -= 0.01f32},
                    VirtualKeyCode::Left => {position.x += 0.01f32},
                    VirtualKeyCode::Right => {position.x -= 0.01f32},
                    _ => ()
                },
                _ => (),
            },
            Event::MainEventsCleared => {
                gvk.set_position(&position);
                gvk.present_loop(&window.window);
            },
            _ => (),
        }
    });
}
