mod pointer_chain_helpers;
mod base_vk;
pub mod graph_vk;

use ash::vk;
use std::fmt::Write;

unsafe extern "system" fn vk_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let mut message = String::new();
    write!(message, "{:?}: ", message_severity).unwrap();
    write!(
        message,
        "[{:?}][{:?}] : {:?}",
        (*p_callback_data).message_id_number,
        (*p_callback_data).p_message_id_name,
        (*p_callback_data).p_message
    )
    .unwrap();
    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        eprintln!("{}", message);
    } else {
        println!("{}", message);
    }
    vk::FALSE
}
