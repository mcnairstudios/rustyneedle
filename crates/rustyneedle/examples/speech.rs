use std::sync::Arc;

use rustyneedle::speech::{Intent, Whistle};
use rustyneedle::{Model, Needle, ToolFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let (Some(tools), Some(clip)) = (arguments.next(), arguments.next()) else {
        eprintln!("usage: speech <tools.json> <clip.wav>");
        std::process::exit(2);
    };

    let whistle = Arc::new(Whistle::open("weights/whistle.cact")?);
    let model = Model::open("weights/needle3.cact")?;
    let agent = Needle::builder().model(&model).tool_file(ToolFile::from_path(tools)?).auto_date(false).build();

    let heard = Intent::new(whistle, agent).process_file(clip)?;
    println!("heard: {}", heard.transcript.text);
    println!("{}", heard.response.to_json());
    Ok(())
}
