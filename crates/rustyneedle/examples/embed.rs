use rustyneedle::{Model, Needle};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::open("weights/needle3.cact")?;
    let agent = Needle::builder().model(&model).build();

    let reference = agent.embed("turn on the kitchen light").ok_or("this model has no embedding head")?;
    for text in ["switch the lamp in the kitchen on", "what is the capital of France", "play some jazz"] {
        let vector = agent.embed(text).ok_or("this model has no embedding head")?;
        println!("{:.3}  {text}", cosine(&reference, &vector));
    }
    Ok(())
}
