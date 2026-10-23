use rustyneedle::{Call, Engine, Needle, Response, ResponseKind};
use serde_json::{json, Map, Value};

struct Scripted {
    answer: Vec<Call>,
}

impl Engine for Scripted {
    fn complete(&mut self, _text: &str, _max_new_tokens: usize) -> Response {
        let mut response = Response::empty(ResponseKind::Call);
        response.function_calls = self.answer.clone();
        response.confidence = Some(0.99);
        response
    }

    fn reset(&mut self) {}

    fn embed(&self, _text: &str) -> Option<Vec<f32>> {
        None
    }
}

fn main() {
    let arguments: Map<String, Value> = json!({"room": "kitchen", "state": "on"}).as_object().cloned().unwrap();
    let engine = Scripted { answer: vec![Call { name: "set_lights".into(), arguments }] };

    let lights = json!({
        "name": "set_lights",
        "description": "Turn the lights in a room on or off.",
        "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "state": {"type": "string"}}, "required": ["room", "state"]}
    });
    let mut agent = Needle::builder()
        .engine(Box::new(engine))
        .tool_with_handler(lights, |arguments| Ok(json!({"done": true, "room": arguments["room"]})))
        .auto_date(false)
        .build();

    let response = agent.run("kitchen lights on", &Default::default());
    println!("{}", response.results.unwrap()[0]);
}
