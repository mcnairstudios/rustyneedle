use rustyneedle::download::{base_weights, Base};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", base_weights(Base::Needle)?.display());
    println!("{}", base_weights(Base::Whistle)?.display());
    Ok(())
}
