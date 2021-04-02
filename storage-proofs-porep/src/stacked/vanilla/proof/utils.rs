use log::*;

const MEMORY_PADDING: f64 = 0.35f64;

pub fn get_memory_padding() -> f64 {
    std::env::var("P2_GPU_MEMORY_PADDING")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid P2_GPU_MEMORY_PADDING! Defaulting to {}", MEMORY_PADDING);
                Ok(MEMORY_PADDING)
            }
        })
        .unwrap_or(MEMORY_PADDING)
        .max(0f64)
        .min(1f64)
}
