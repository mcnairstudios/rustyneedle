const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

pub fn prefix(system: &str, tools_json: &str) -> String {
    let system_turn = if system.is_empty() { String::new() } else { format!("{IM_START}system\n{system}{IM_END}\n") };
    format!("{system_turn}{IM_START}user\n<tools>{tools_json}</tools>")
}

pub fn first_user_turn(query: &str) -> String {
    format!("\n{query}{IM_END}\n{IM_START}assistant\n")
}

pub fn later_user_turn(query: &str) -> String {
    format!("\n{IM_START}user\n{query}{IM_END}\n{IM_START}assistant\n")
}

pub fn tool_result_turn(result_json: &str) -> String {
    format!("\n{IM_START}tool\n<tool_result>{result_json}</tool_result>{IM_END}\n{IM_START}assistant\n")
}
