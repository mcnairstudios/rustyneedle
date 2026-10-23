use needle::Needle;
use rustyneedle_intent::{schema_keywords, Intent};
use serde::Deserialize;
use serde_json::Value;
use whistle::Whistle;

#[derive(Deserialize)]
struct Clip {
    clip: String,
    spoken: String,
}

const ENVIRONMENTS: [&str; 6] = ["data_capture", "kitchen_appliance", "media_player", "productivity", "smart_home", "wearable"];

fn names(calls: &Value) -> Vec<String> {
    calls.as_array().into_iter().flatten().filter_map(|call| call["name"].as_str().map(str::to_owned)).collect()
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn spoken_requests_reach_the_same_tools_as_their_text() {
    let model = model::Model::open(testkit::workspace_file("weights/needle3.cact")).unwrap();
    let goldens: Vec<Value> = testkit::golden("environments");
    let clips: Vec<Clip> = testkit::golden("speech");
    let whistle = std::sync::Arc::new(Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap());
    let mut agreed = 0;
    let mut report = Vec::new();
    for (index, clip) in clips.iter().enumerate() {
        let environment = ENVIRONMENTS[index / 4];
        let meta = goldens.iter().find(|g| g.get("tool_schemas").is_some() && g["environment"] == environment).unwrap();
        let tools = meta["tool_schemas"].as_array().cloned().unwrap();
        let agent = Needle::builder().model(&model).tools(tools).system(meta["system"].as_str().unwrap()).build();
        let mut pipeline = Intent::new(std::sync::Arc::clone(&whistle), agent);
        let spoken = pipeline.process_file(testkit::workspace_file(&format!("oracle/clips/{}", clip.clip))).unwrap();
        let want = goldens.iter().find(|g| g["environment"] == environment && g["query"] == clip.spoken.as_str()).unwrap();
        let got = spoken.to_json();
        assert!(got["audio_text"].is_string() && got["audio_language"] == "en");
        let calls = |value: &Value| {
            let mut all = names(&value["function_calls"]);
            all.extend(names(&value["suppressed_calls"]));
            all
        };
        if calls(&got) == calls(&want["response"]) {
            agreed += 1;
        } else {
            report.push(format!("{} heard {:?}: {:?} vs {:?}", clip.spoken, got["audio_text"], calls(&got), calls(&want["response"])));
        }
    }
    println!("{}\n{agreed}/{} spoken requests reached the same tools", report.join("\n"), clips.len());
    assert!(agreed * 10 >= clips.len() * 7);
}

#[test]
fn schema_enum_values_become_keywords() {
    let tools = vec![serde_json::json!({"name": "set_lights", "parameters": {"type": "object", "properties": {
        "room": {"type": "string", "enum": ["kitchen", "living_room"]}, "state": {"type": "string", "enum": ["on", "off"]}}}})];
    assert_eq!(schema_keywords(&tools), vec!["kitchen", "living room", "off", "on"]);
}
