const TERNARY_LEVEL: f32 = 1.224_006_4;

#[derive(Clone, Debug)]
pub struct Codebooks {
    two_bit: [f32; 4],
    three_bit: [f32; 8],
    four_bit: [f32; 16],
    group: usize,
}

impl Codebooks {
    pub fn from_archive(values: &[f32], group: usize) -> Self {
        let mut two_bit = [0.0; 4];
        let mut three_bit = [0.0; 8];
        let mut four_bit = [0.0; 16];
        two_bit.copy_from_slice(&values[0..4]);
        three_bit.copy_from_slice(&values[4..12]);
        four_bit.copy_from_slice(&values[12..28]);
        Self { two_bit, three_bit, four_bit, group }
    }

    pub fn two_bit(&self) -> &[f32; 4] {
        &self.two_bit
    }

    pub fn three_bit(&self) -> &[f32; 8] {
        &self.three_bit
    }

    pub fn four_bit(&self) -> &[f32; 16] {
        &self.four_bit
    }

    pub fn binary(&self) -> [f32; 2] {
        let level = (2.0 / std::f32::consts::PI).sqrt() / (self.group as f32).sqrt();
        [-level, level]
    }

    pub fn ternary_by_crumb(&self) -> [f32; 4] {
        let level = TERNARY_LEVEL / (self.group as f32).sqrt();
        [0.0, level, 0.0, -level]
    }
}
