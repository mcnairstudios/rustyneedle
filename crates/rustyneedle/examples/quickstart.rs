use rustyneedle::{tool, Model, Needle, Parameter, RunOptions};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::open("weights/needle3.cact")?;

    let weather = tool("get_weather").description("Get the current weather for a city.").parameter(Parameter::string("city")).build();

    let mut agent = Needle::builder().model(&model).tool_with_handler(weather, |arguments| Ok(json!({"city": arguments["city"], "temp_c": 27}))).build();

    let response = agent.run("what's it like in Lagos right now?", &RunOptions::default());
    println!("{}", response.results.unwrap()[0]);
    Ok(())
}
