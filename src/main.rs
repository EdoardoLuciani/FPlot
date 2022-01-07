mod renderer;

use renderer::vk::graph_vk::GraphVk;
use crate::renderer::window_manager::WindowManager;

use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event::WindowEvent::KeyboardInput;
use winit::event_loop::ControlFlow;
use winit::platform::run_return::EventLoopExtRunReturn;

fn main() {
    std::env::set_var("WINIT_UNIX_BACKEND", "x11");
    let mut window = WindowManager::new((800u32, 800u32), None);
    let mut gvk = GraphVk::new((800u32, 800u32), window.get_window_handle());
    gvk.fill_graph_buffer(|x| x);
    gvk.prepare();
    window.event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            Event::MainEventsCleared => gvk.present_loop(&window.window),
            _ => (),
        }
    });
}
