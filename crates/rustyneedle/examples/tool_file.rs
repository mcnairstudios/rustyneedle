use rustyneedle::{Model, Needle, ToolFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let (Some(tools), Some(request)) = (arguments.next(), arguments.next()) else {
        eprintln!("usage: tool_file <tools.json> <request>");
        std::process::exit(2);
    };

    let model = Model::open("weights/needle3.cact")?;
    let mut agent = Needle::builder().model(&model).tool_file(ToolFile::from_path(tools)?).auto_date(false).build();

    let response = agent.complete(&request);
    for call in &response.function_calls {
        println!("{} {}", call.name, serde_json::Value::Object(call.arguments.clone()));
    }
    if response.function_calls.is_empty() {
        println!("no call");
    }
    if let Some(confidence) = response.confidence {
        println!("confidence {confidence:.3}");
    }
    Ok(())
}
