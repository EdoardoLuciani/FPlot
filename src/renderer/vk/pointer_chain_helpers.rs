use ash::vk;
use std::alloc::Layout;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::null_mut;

pub unsafe fn clone_vkPhysicalDeviceFeatures2_structure(
    source: &vk::PhysicalDeviceFeatures2,
) -> vk::PhysicalDeviceFeatures2 {
    let mut ret_val = vk::PhysicalDeviceFeatures2::default();

    let mut source_ptr = source.p_next;
    let mut dst_ptr = &mut (ret_val.p_next);
    while !source_ptr.is_null() {
        let cloned_child_struct_ptr;
        match (*(source_ptr as *const vk::PhysicalDeviceFeatures2)).s_type {
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => {
                let lay = Layout::new::<vk::PhysicalDeviceVulkan11Features>();
                cloned_child_struct_ptr = std::alloc::alloc(lay);
                (*(cloned_child_struct_ptr as *mut vk::PhysicalDeviceVulkan11Features)).s_type =
                    vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
            }
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT => {
                let lay = Layout::new::<vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>();
                cloned_child_struct_ptr = std::alloc::alloc(lay);
                (*(cloned_child_struct_ptr as *mut vk::PhysicalDeviceVulkan11Features)).s_type =
                    vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
            }
            _ => panic!("Found unrecognized struct inside clone_vkPhysicalDeviceFeatures2"),
        }
        (*(cloned_child_struct_ptr as *mut vk::PhysicalDeviceVulkan11Features)).p_next = null_mut();
        *dst_ptr = cloned_child_struct_ptr as *mut c_void;
        dst_ptr = &mut ((*((*dst_ptr) as *mut vk::PhysicalDeviceFeatures2)).p_next);
        source_ptr = (*(source_ptr as *const vk::PhysicalDeviceFeatures2)).p_next;
    }
    ret_val
}

pub unsafe fn destroy_vkPhysicalDeviceFeatures2(source: &mut vk::PhysicalDeviceFeatures2) {
    let mut p_next = source.p_next;
    let mut p_next_tmp;
    while !p_next.is_null() {
        match (*(p_next as *const vk::PhysicalDeviceFeatures2)).s_type {
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => {
                p_next_tmp = p_next;
                p_next = (*(p_next as *const vk::PhysicalDeviceFeatures2)).p_next;
                std::alloc::dealloc(
                    p_next_tmp as *mut u8,
                    Layout::new::<vk::PhysicalDeviceVulkan11Features>(),
                );
            }
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT => {
                p_next_tmp = p_next;
                p_next = (*(p_next as *const vk::PhysicalDeviceFeatures2)).p_next;
                std::alloc::dealloc(
                    p_next_tmp as *mut u8,
                    Layout::new::<vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>(),
                );
            }
            _ => panic!("Found unrecognized struct inside destroy_vkPhysicalDeviceFeatures2"),
        }
    }
    source.p_next = null_mut();
}

pub unsafe fn compare_device_features_structs(
    baseline: *const c_void,
    desired: *const c_void,
    size: usize,
) -> bool {
    // casting the structure to a PhysicalDeviceFeatures2 struct to compare the struct identifier
    if (*(baseline as *const vk::PhysicalDeviceFeatures2)).s_type
        != (*(desired as *const vk::PhysicalDeviceFeatures2)).s_type
    {
        return false;
    }

    // then we know that the structs type are the same so we cast them to a view of u32
    // the offset has a 4 added to it because of struct padding
    let offset = size_of::<ash::vk::StructureType>() + size_of::<*mut c_void>() + 4;
    let baseline_data = baseline.add(offset) as *const u32;
    let desired_data = desired.add(offset) as *const u32;
    for i in 0..((size - offset) / size_of::<u32>()) {
        if *(desired_data.add(i)) > *(baseline_data.add(i)) {
            return false;
        }
    }
    true
}

pub unsafe fn compare_vkPhysicalDeviceFeatures2(
    baseline: &vk::PhysicalDeviceFeatures2,
    desired: &vk::PhysicalDeviceFeatures2,
) -> bool {
    if !compare_device_features_structs(
        baseline as *const vk::PhysicalDeviceFeatures2 as *const c_void,
        desired as *const vk::PhysicalDeviceFeatures2 as *const c_void,
        size_of::<vk::PhysicalDeviceFeatures2>(),
    ) {
        return false;
    }

    let mut baseline_ptr = baseline.p_next;
    let mut desired_ptr = desired.p_next;
    while !baseline_ptr.is_null() && !desired_ptr.is_null() {
        let res: bool;
        match (*(baseline_ptr as *const vk::PhysicalDeviceFeatures2)).s_type {
            vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_1_FEATURES => {
                res = compare_device_features_structs(
                    baseline_ptr,
                    desired_ptr,
                    size_of::<vk::PhysicalDeviceVulkan11Features>(),
                );
            }
            vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT => {
                res = compare_device_features_structs(
                    baseline_ptr,
                    desired_ptr,
                    size_of::<vk::PhysicalDeviceDescriptorIndexingFeaturesEXT>(),
                );
            }
            _ => panic!("Found unrecognized struct inside compare_vkPhysicalDeviceFeatures2"),
        }
        if !res {
            return false;
        }
        baseline_ptr = (*(baseline_ptr as *const vk::PhysicalDeviceFeatures2)).p_next;
        desired_ptr = (*(desired_ptr as *const vk::PhysicalDeviceFeatures2)).p_next;
    }
    baseline_ptr.is_null() && desired_ptr.is_null()
}
