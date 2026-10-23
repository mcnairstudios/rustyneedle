use rustyneedle_needle::{to_snake_case, tool, Parameter, ToolSet};
use serde_json::{json, Value};

fn required(schema: &Value) -> Vec<String> {
    schema["parameters"]["required"].as_array().map(|list| list.iter().map(|v| v.as_str().unwrap().to_owned()).collect()).unwrap_or_default()
}

#[test]
fn basic_schema_types_and_required() {
    let schema = tool("add").description("Add two numbers.").parameter(Parameter::integer("a")).parameter(Parameter::integer("b")).build();
    assert_eq!(schema["name"], "add");
    assert_eq!(schema["description"], "Add two numbers.");
    assert_eq!(schema["parameters"]["properties"]["a"], json!({"type": "integer"}));
    assert_eq!(required(&schema), vec!["a", "b"]);
}

#[test]
fn optional_parameters_are_not_required_and_an_empty_required_list_is_left_out() {
    let schema = tool("f").parameter(Parameter::string("city")).parameter(Parameter::string("units").optional()).build();
    assert_eq!(required(&schema), vec!["city"]);
    assert!(schema["parameters"]["properties"]["units"].is_object());
    let none = tool("g").parameter(Parameter::string("x").optional()).build();
    assert!(none["parameters"].get("required").is_none());
}

#[test]
fn choices_lists_objects_and_typed_items() {
    let schema = tool("f")
        .parameter(Parameter::string("mode").choices(["fast", "slow"]))
        .parameter(Parameter::array_of("tags", Parameter::integer("item")))
        .parameter(Parameter::object("meta"))
        .build();
    let properties = &schema["parameters"]["properties"];
    assert_eq!(properties["mode"], json!({"type": "string", "enum": ["fast", "slow"]}));
    assert_eq!(properties["tags"], json!({"type": "array", "items": {"type": "integer"}}));
    assert_eq!(properties["meta"], json!({"type": "object"}));
}

#[test]
fn numeric_choices_take_a_numeric_type() {
    let schema =
        tool("f").parameter(Parameter::integer("port").choices([80, 443, 8080])).parameter(Parameter::string("mode").choices(["fast"]).optional()).build();
    assert_eq!(schema["parameters"]["properties"]["port"], json!({"type": "integer", "enum": [80, 443, 8080]}));
}

#[test]
fn field_constraints_map_onto_schema_keywords() {
    let schema = tool("f")
        .parameter(Parameter::integer("temp").description("temperature").at_least(0).at_most(100))
        .parameter(Parameter::string("name").optional().pattern("^[a-z]+$").min_length(2).max_length(9).format("email"))
        .parameter(Parameter::number("amount").above(0).below(10).multiple_of(0.5))
        .parameter(Parameter::array_of("items", Parameter::string("item")).min_items(1).max_items(3).unique_items(true))
        .build();
    let properties = &schema["parameters"]["properties"];
    assert_eq!(properties["temp"], json!({"type": "integer", "description": "temperature", "minimum": 0, "maximum": 100}));
    assert_eq!(properties["name"], json!({"type": "string", "pattern": "^[a-z]+$", "minLength": 2, "maxLength": 9, "format": "email"}));
    assert_eq!(properties["amount"], json!({"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 10, "multipleOf": 0.5}));
    assert_eq!(properties["items"]["minItems"], 1);
    assert_eq!(properties["items"]["uniqueItems"], true);
    assert_eq!(required(&schema), vec!["temp", "amount", "items"]);
}

#[test]
fn triggers_ride_beside_the_schema() {
    let schema = tool("toggle").triggers(["\\btoggle\\b"]).build();
    assert_eq!(schema["triggers"], json!(["\\btoggle\\b"]));
}

#[test]
fn names_become_snake_case_and_calls_map_back() {
    assert_eq!(to_snake_case("setLights"), "set_lights");
    assert_eq!(to_snake_case("Set Lights"), "set_lights");
    assert_eq!(to_snake_case("media-player__playTrack"), "media_player_play_track");
    let tools = json!([{"type": "function", "function": {"name": "setLights", "description": "d", "parameters": {"type": "object", "properties": {"roomName": {"type": "string"}}, "required": ["roomName"]}, "triggers": ["x"]}}]);
    let set = ToolSet::new(&tools);
    let prompt = set.prompt_tools();
    assert_eq!(prompt[0]["name"], "set_lights");
    assert!(prompt[0].get("triggers").is_none());
    assert!(prompt[0]["parameters"]["properties"].get("room_name").is_some());
    assert_eq!(prompt[0]["parameters"]["required"], json!(["room_name"]));
    assert_eq!(set.original_name("set_lights"), "setLights");
    let restored = set.original_arguments("set_lights", json!({"room_name": "study"}).as_object().unwrap());
    assert_eq!(restored["roomName"], "study");
}

#[test]
fn triggers_select_the_tools_whose_pattern_matches_the_request() {
    let tools = json!([
        {"name": "toggle", "parameters": {"type": "object", "properties": {}}, "triggers": ["\\b(turn|switch)\\b.*\\b(on|off)\\b"]},
        {"name": "other", "parameters": {"type": "object", "properties": {}}}
    ]);
    let set = ToolSet::new(&tools);
    assert_eq!(set.triggered("Switch the fan ON"), Some(vec![0]));
    assert_eq!(set.triggered("what time is it"), None);
}
