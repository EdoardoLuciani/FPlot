use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use winit::dpi::{LogicalSize, Size};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::Fullscreen;
use winit::*;

pub struct WindowManager {
    event_loop: EventLoop<()>,
    window: window::Window,
}

impl WindowManager {
    pub fn new(resolution: (u32, u32), fullscreen: Option<Fullscreen>) -> Self {
        let event_loop = event_loop::EventLoop::new();
        let window = window::WindowBuilder::new()
            .with_fullscreen(fullscreen)
            .with_min_inner_size(LogicalSize {
                width: resolution.0,
                height: resolution.1,
            })
            .build(&event_loop)
            .unwrap();
        WindowManager { event_loop, window }
    }

    pub fn get_window_handle(&self) -> RawWindowHandle {
        self.window.raw_window_handle()
    }

    pub fn start(&mut self) {
        self.event_loop.run_return(|event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == self.window.id() => *control_flow = ControlFlow::Exit,
                _ => (),
            }
        });
    }
}
