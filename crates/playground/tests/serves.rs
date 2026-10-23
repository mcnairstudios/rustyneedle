use rustyneedle_playground::{Config, Playground};
use serde_json::{json, Value};

fn start() -> (String, std::sync::mpsc::Sender<()>, std::thread::JoinHandle<()>) {
    let config = Config { host: "127.0.0.1".into(), port: 0, model: testkit::workspace_file("weights/needle3.cact"), depth: None };
    let playground = Playground::bind(&config).unwrap();
    let address = playground.address().unwrap();
    let (stop, receive) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || playground.run_until(receive));
    (format!("http://{address}"), stop, handle)
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn the_playground_serves_its_page_and_answers_tool_calls() {
    let (base, stop, handle) = start();
    let page = ureq::get(&format!("{base}/")).call().unwrap().into_string().unwrap();
    assert!(page.contains("<html") || page.contains("<!DOCTYPE") || page.contains("<!doctype"));
    assert!(ureq::get(&format!("{base}/app.js")).call().unwrap().into_string().unwrap().contains("fetch("));
    let model: Value = ureq::get(&format!("{base}/model")).call().unwrap().into_json().unwrap();
    assert_eq!(model["name"], "needle3.cact");
    let tools = json!([{"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]);
    let answer: Value =
        ureq::post(&format!("{base}/complete")).send_json(json!({"tools": tools, "query": "what's the weather in Lagos?"})).unwrap().into_json().unwrap();
    assert_eq!(answer["function_calls"][0]["name"], "get_weather");
    assert_eq!(answer["function_calls"][0]["arguments"]["city"], "Lagos");
    let reset: Value = ureq::post(&format!("{base}/reset")).call().unwrap().into_json().unwrap();
    assert_eq!(reset["ok"], true);
    let status = ureq::get(&format!("{base}/nothing")).call();
    assert!(matches!(status, Err(ureq::Error::Status(404, _))));
    stop.send(()).unwrap();
    handle.join().unwrap();
}
