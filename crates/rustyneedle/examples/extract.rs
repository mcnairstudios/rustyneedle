use rustyneedle::{extract, ExtractOptions, Model};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::open("weights/needle3.cact")?;

    let booking = json!({
        "name": "booking",
        "description": "A restaurant booking.",
        "parameters": {
            "type": "object",
            "properties": {
                "restaurant": {"type": "string"},
                "party_size": {"type": "integer"}
            },
            "required": ["restaurant", "party_size"]
        }
    });

    let text = "Book a table for four at Luigi's on Friday";
    match extract(&model, booking, text, &ExtractOptions::default())? {
        Some(record) => println!("{}", serde_json::Value::Object(record)),
        None => println!("nothing to extract"),
    }
    Ok(())
}
