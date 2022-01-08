mod renderer;

use std::time::{Duration, Instant};
use renderer::vk::graph_vk::GraphVk;
use crate::renderer::window_manager::WindowManager;

use winit::event::*;
use winit::event_loop::ControlFlow;
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::dpi::PhysicalPosition;

use nalgebra::*;

fn main() {
    std::env::set_var("WINIT_UNIX_BACKEND", "x11");
    let mut window = WindowManager::new((800u32, 800u32), None);
    let mut gvk = GraphVk::new((800u32, 800u32), window.get_window_handle());
    let mut center = Vector3::new(0.0f32, 0.0f32, 0.0f32);

    let mut left_mouse_pressed = false;
    let mut last_mouse_pressed_pos: Option<PhysicalPosition<f64>> = None;
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
                WindowEvent::MouseInput {
                    state: pressed_state,
                    button: MouseButton::Left,
                    ..
                } => left_mouse_pressed = match pressed_state {
                    ElementState::Pressed => true,
                    ElementState::Released => {
                        last_mouse_pressed_pos = None;
                        false
                    },
                },
                WindowEvent::CursorMoved {
                    position: pos,
                    ..
                } => {
                    if last_mouse_pressed_pos.is_some() && left_mouse_pressed {
                        let ex_pos_vec = Vector2::new(last_mouse_pressed_pos.unwrap().x, last_mouse_pressed_pos.unwrap().y);
                        let pos_vec = Vector2::new(pos.x, pos.y);
                        let window_size_vec = Vector2::new(window.window.inner_size().width as f64, window.window.inner_size().height as f64);

                        let delta_pos = (ex_pos_vec - pos_vec);
                        let delta_pos_normalized = delta_pos.component_div(&window_size_vec);
                        center -= Vector3::new(delta_pos_normalized.x as f32, delta_pos_normalized.y as f32, 0.0f32);
                        window.window.request_redraw();
                    }
                    last_mouse_pressed_pos = Some(pos);
                },
                _ => (),
            },
            Event::RedrawRequested(_) => {
                gvk.fill_graph_buffer(-center.x - 1.0f32, -center.x + 1.0f32, |x| f32::sin(4.0f32*x));
                gvk.set_position(&center);
                gvk.present_loop(&window.window);
            },
            _ => (),
        }
    });
}
