use std::sync::Arc;

use serde_json::Value;

const MAX_DEPTH: usize = 12;
const MAX_ARRAY_ITEMS: usize = 64;
const DEFAULT_ARRAY_MAX: usize = MAX_ARRAY_ITEMS;

#[derive(Clone, Debug)]
pub struct Property {
    pub name: String,
    pub node: Node,
    pub required: bool,
}

#[derive(Clone, Debug)]
pub enum Node {
    Text { min_chars: usize, max_chars: Option<usize> },
    Integer { min: Option<i64>, max: Option<i64> },
    Number { min: Option<f64>, max: Option<f64> },
    Boolean,
    Null,
    Literals(Arc<[String]>),
    Array { items: Arc<Node>, min_items: usize, max_items: usize },
    Object { properties: Arc<[Property]> },
    Choice(Arc<[Node]>),
}

pub fn compile(schema: &Value, root: &Value) -> Node {
    compile_at(schema, root, 0)
}

fn resolve<'a>(schema: &'a Value, root: &'a Value) -> &'a Value {
    let mut current = schema;
    for _ in 0..MAX_DEPTH {
        let Some(reference) = current.get("$ref").and_then(Value::as_str) else { return current };
        let Some(path) = reference.strip_prefix("#/") else { return current };
        let mut target = root;
        for part in path.split('/') {
            match target.get(part) {
                Some(next) => target = next,
                None => return current,
            }
        }
        current = target;
    }
    current
}

fn literal_text(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_default()
}

fn compile_at(schema: &Value, root: &Value, depth: usize) -> Node {
    let schema = resolve(schema, root);
    if depth > MAX_DEPTH {
        return Node::Text { min_chars: 0, max_chars: None };
    }
    if let Some(constant) = schema.get("const") {
        return Node::Literals(vec![literal_text(constant)].into());
    }
    if let Some(options) = schema.get("enum").and_then(Value::as_array) {
        return Node::Literals(options.iter().map(literal_text).collect());
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(variants) = schema.get(key).and_then(Value::as_array) {
            let nodes: Vec<Node> = variants.iter().map(|variant| compile_at(variant, root, depth + 1)).collect();
            return if nodes.len() == 1 { nodes.into_iter().next().expect("one variant") } else { Node::Choice(nodes.into()) };
        }
    }
    if let Some(kinds) = schema.get("type").and_then(Value::as_array) {
        let nodes: Vec<Node> = kinds
            .iter()
            .filter_map(Value::as_str)
            .map(|kind| {
                let mut single = schema.clone();
                single["type"] = Value::String(kind.to_owned());
                compile_at(&single, root, depth + 1)
            })
            .collect();
        return Node::Choice(nodes.into());
    }
    match schema.get("type").and_then(Value::as_str).unwrap_or("string") {
        "integer" => integer_node(schema),
        "number" => number_node(schema),
        "boolean" => Node::Boolean,
        "null" => Node::Null,
        "array" => array_node(schema, root, depth),
        "object" => object_node(schema, root, depth),
        _ => text_node(schema),
    }
}

fn text_node(schema: &Value) -> Node {
    let min_chars = schema.get("minLength").and_then(Value::as_u64).unwrap_or(0) as usize;
    let max_chars = schema.get("maxLength").and_then(Value::as_u64).map(|value| value as usize);
    Node::Text { min_chars, max_chars }
}

fn integer_node(schema: &Value) -> Node {
    let lower = schema.get("minimum").and_then(Value::as_f64).map(|v| v.ceil() as i64);
    let upper = schema.get("maximum").and_then(Value::as_f64).map(|v| v.floor() as i64);
    let lower_exclusive = schema.get("exclusiveMinimum").and_then(Value::as_f64).map(|v| v.floor() as i64 + 1);
    let upper_exclusive = schema.get("exclusiveMaximum").and_then(Value::as_f64).map(|v| v.ceil() as i64 - 1);
    Node::Integer { min: tighter(lower, lower_exclusive, i64::max), max: tighter(upper, upper_exclusive, i64::min) }
}

fn number_node(schema: &Value) -> Node {
    let min = schema.get("minimum").and_then(Value::as_f64).or_else(|| schema.get("exclusiveMinimum").and_then(Value::as_f64));
    let max = schema.get("maximum").and_then(Value::as_f64).or_else(|| schema.get("exclusiveMaximum").and_then(Value::as_f64));
    Node::Number { min, max }
}

fn tighter(a: Option<i64>, b: Option<i64>, pick: fn(i64, i64) -> i64) -> Option<i64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(pick(x, y)),
        (x, None) => x,
        (None, y) => y,
    }
}

fn array_node(schema: &Value, root: &Value, depth: usize) -> Node {
    let items = schema.get("items").map(|items| compile_at(items, root, depth + 1)).unwrap_or(Node::Text { min_chars: 0, max_chars: None });
    let min_items = schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
    let max_items = schema.get("maxItems").and_then(Value::as_u64).map_or(DEFAULT_ARRAY_MAX, |v| (v as usize).min(MAX_ARRAY_ITEMS));
    Node::Array { items: Arc::new(items), min_items, max_items }
}

fn object_node(schema: &Value, root: &Value, depth: usize) -> Node {
    let required: Vec<&str> = schema.get("required").and_then(Value::as_array).map(|list| list.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let properties: Vec<Property> = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|map| {
            map.iter()
                .map(|(name, child)| Property { name: name.clone(), node: compile_at(child, root, depth + 1), required: required.contains(&name.as_str()) })
                .collect()
        })
        .unwrap_or_default();
    Node::Object { properties: Arc::from(properties) }
}
