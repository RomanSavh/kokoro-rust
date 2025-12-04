use std::time::Instant;

const COLORS: &[&str] = &[
    "\x1b[31m", "\x1b[32m", "\x1b[33m", "\x1b[34m", "\x1b[35m", "\x1b[36m", "\x1b[91m", "\x1b[92m",
    "\x1b[93m", "\x1b[94m", "\x1b[95m", "\x1b[96m", "\x1b[37m", "\x1b[90m",
];
const RESET: &str = "\x1b[0m";

/// Get consistent color for a request ID using hash-based assignment
pub fn get_request_id_color(request_id: &str) -> &'static str {
    let mut hash = 0u32;
    for byte in request_id.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
    }
    let color_index = (hash as usize) % COLORS.len();
    COLORS[color_index]
}

pub fn format_debug_prefix(request_id: Option<&str>, instance_id: Option<&str>) -> String {
    match (request_id, instance_id) {
        (Some(req_id), Some(inst_id)) => {
            let color = get_request_id_color(req_id);
            format!("{color}[{req_id}]{RESET}[{inst_id}]")
        }
        (Some(req_id), None) => {
            let color = get_request_id_color(req_id);
            format!("{color}[{req_id}]{RESET}")
        }
        (None, Some(inst_id)) => format!("[{inst_id}]"),
        (None, None) => String::new(),
    }
}

pub fn get_colored_request_id_with_relative(request_id: &str, start_time: Instant) -> String {
    let color = get_request_id_color(request_id);

    // Get relative time from request start
    let elapsed_ms = start_time.elapsed().as_millis();
    let relative_time = if elapsed_ms < 1 {
        "    0".to_string() // Show "0" right-aligned for initial request
    } else {
        format!("{elapsed_ms:5}") // Right-aligned 5 digits
    };

    format!("{color}[{request_id}]{RESET} \x1b[90m{relative_time}\x1b[0m")
}

pub fn pcm_f32_to_i16(input: &[f32]) -> Vec<i16> {
    input
        .iter()
        .map(|&s| {
            let s = s.clamp(-1.0, 1.0); // запобігаємо переповненню
            (s * i16::MAX as f32) as i16
        })
        .collect()
}
