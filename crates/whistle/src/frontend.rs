use std::f32::consts::PI;

const FLOOR: f32 = 5.960_464_5e-8;
const NORMALISER_EPSILON: f32 = 1e-5;

pub struct FrontendConfig {
    pub mels: usize,
    pub fft_size: usize,
    pub window: usize,
    pub hop: usize,
}

pub struct Features {
    pub spread: f32,
    pub data: Vec<f32>,
    pub frames: usize,
    pub valid: usize,
    pub mels: usize,
}

pub struct Frontend {
    config: FrontendConfig,
    window: Vec<f32>,
    mel: Vec<f32>,
    twiddle_cos: Vec<f32>,
    twiddle_sin: Vec<f32>,
    bit_reverse: Vec<usize>,
}

impl Frontend {
    pub fn new(config: FrontendConfig, mel: Vec<f32>) -> Self {
        let size = config.fft_size;
        let window = (0..config.window).map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / (config.window - 1) as f32).cos()).collect();
        let twiddle_cos = (0..size / 2).map(|k| (2.0 * PI * k as f32 / size as f32).cos()).collect();
        let twiddle_sin = (0..size / 2).map(|k| -(2.0 * PI * k as f32 / size as f32).sin()).collect();
        let bits = size.trailing_zeros();
        let bit_reverse = (0..size).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect();
        Self { config, window, mel, twiddle_cos, twiddle_sin, bit_reverse }
    }

    fn power_spectrum(&self, real: &mut [f32], imaginary: &mut [f32], output: &mut [f32]) {
        let size = self.config.fft_size;
        for index in 0..size {
            let target = self.bit_reverse[index];
            if target > index {
                real.swap(index, target);
                imaginary.swap(index, target);
            }
        }
        let mut span = 1;
        while span < size {
            let stride = size / (span * 2);
            for start in (0..size).step_by(span * 2) {
                for offset in 0..span {
                    let (cos, sin) = (self.twiddle_cos[offset * stride], self.twiddle_sin[offset * stride]);
                    let (a, b) = (start + offset, start + offset + span);
                    let tr = real[b] * cos - imaginary[b] * sin;
                    let ti = real[b] * sin + imaginary[b] * cos;
                    real[b] = real[a] - tr;
                    imaginary[b] = imaginary[a] - ti;
                    real[a] += tr;
                    imaginary[a] += ti;
                }
            }
            span *= 2;
        }
        for bin in 0..=size / 2 {
            output[bin] = real[bin] * real[bin] + imaginary[bin] * imaginary[bin];
        }
    }

    pub fn log_mel(&self, samples: &[f32]) -> Features {
        let config = &self.config;
        let frames = samples.len() / config.hop + 1;
        let valid = samples.len() / config.hop;
        let half = config.window / 2;
        let bins = config.fft_size / 2 + 1;
        let mut data = vec![0.0f32; frames * config.mels];
        let mut real = vec![0.0f32; config.fft_size];
        let mut imaginary = vec![0.0f32; config.fft_size];
        let mut power = vec![0.0f32; bins];
        for frame in 0..frames {
            real.iter_mut().for_each(|v| *v = 0.0);
            imaginary.iter_mut().for_each(|v| *v = 0.0);
            for (position, (slot, gain)) in real.iter_mut().zip(&self.window).enumerate() {
                let source = (frame * config.hop + position) as isize - half as isize;
                if source >= 0 && (source as usize) < samples.len() {
                    *slot = samples[source as usize] * gain;
                }
            }
            self.power_spectrum(&mut real, &mut imaginary, &mut power);
            let row = &mut data[frame * config.mels..(frame + 1) * config.mels];
            for (bin, value) in power.iter().enumerate() {
                let weights = &self.mel[bin * config.mels..(bin + 1) * config.mels];
                for (slot, weight) in row.iter_mut().zip(weights) {
                    *slot += value * weight;
                }
            }
            row.iter_mut().for_each(|value| *value = (*value + FLOOR).ln());
        }
        let spread = self.spread(&data, valid);
        self.normalise(&mut data, valid);
        Features { spread, data, frames, valid, mels: config.mels }
    }

    fn spread(&self, data: &[f32], valid: usize) -> f32 {
        let mels = self.config.mels;
        if valid < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for mel in 0..mels {
            let mean = (0..valid).map(|frame| data[frame * mels + mel]).sum::<f32>() / valid as f32;
            let variance = (0..valid).map(|frame| (data[frame * mels + mel] - mean).powi(2)).sum::<f32>() / (valid - 1) as f32;
            total += variance.sqrt();
        }
        total / mels as f32
    }

    fn normalise(&self, data: &mut [f32], valid: usize) {
        let mels = self.config.mels;
        let count = valid.max(1) as f32;
        let denominator = (valid.max(1) as f32 - 1.0).max(1.0);
        for mel in 0..mels {
            let mean = (0..valid).map(|frame| data[frame * mels + mel]).sum::<f32>() / count;
            let variance = (0..valid).map(|frame| (data[frame * mels + mel] - mean).powi(2)).sum::<f32>() / denominator;
            let scale = 1.0 / (variance.sqrt() + NORMALISER_EPSILON);
            for frame in 0..data.len() / mels {
                let slot = &mut data[frame * mels + mel];
                *slot = if frame < valid { (*slot - mean) * scale } else { 0.0 };
            }
        }
    }
}
