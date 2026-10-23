use cact::{Archive, StemTensors};
use kernels::{fake_quantise_int8, silu, QuantisedMatrix};
use model::tensors::floats;

const KERNEL: usize = 3;
const STRIDE: usize = 2;

pub struct Stem {
    channels: usize,
    first: Vec<f32>,
    stages: Vec<(Vec<f32>, QuantisedMatrix)>,
    out: QuantisedMatrix,
}

pub struct StemOutput {
    pub data: Vec<f32>,
    pub frames: usize,
    pub valid: usize,
}

fn reduced(length: usize) -> usize {
    (length.max(1) - 1) / STRIDE + 1
}

impl Stem {
    pub fn load(archive: &Archive, tensors: &StemTensors, channels: usize) -> Self {
        Self {
            channels,
            first: floats(archive, tensors.first),
            stages: tensors
                .stages
                .iter()
                .map(|(depthwise, pointwise)| (floats(archive, *depthwise), QuantisedMatrix::from_archive(archive, *pointwise)))
                .collect(),
            out: QuantisedMatrix::from_archive(archive, tensors.out),
        }
    }

    pub fn forward(&self, features: &[f32], frames: usize, valid: usize, mels: usize) -> StemOutput {
        let channels = self.channels;
        let mut length = valid;
        let (mut time, mut freq) = (reduced(frames), reduced(mels));
        length = length.div_ceil(2);
        let mut current = vec![0.0f32; time * freq * channels];
        kernels::parallel_chunks_mut(&mut current, freq * channels, &|t, row| {
            for f in 0..freq {
                let slot = &mut row[f * channels..(f + 1) * channels];
                for dt in 0..KERNEL {
                    for df in 0..KERNEL {
                        let (source_t, source_f) = ((t * STRIDE + dt) as isize - 1, (f * STRIDE + df) as isize - 1);
                        if source_t < 0 || source_f < 0 || source_t as usize >= frames || source_f as usize >= mels {
                            continue;
                        }
                        let value = features[source_t as usize * mels + source_f as usize];
                        let weights = &self.first[(dt * KERNEL + df) * channels..(dt * KERNEL + df + 1) * channels];
                        for (out, weight) in slot.iter_mut().zip(weights) {
                            *out += value * weight;
                        }
                    }
                }
            }
        });
        mask_rows(&mut current, time, freq * channels, length);
        for (depthwise, pointwise) in &self.stages {
            current.iter_mut().for_each(|value| *value = silu(*value));
            let (next_time, next_freq) = (reduced(time), reduced(freq));
            length = length.div_ceil(2);
            let mut downsampled = vec![0.0f32; next_time * next_freq * channels];
            kernels::parallel_chunks_mut(&mut downsampled, next_freq * channels, &|t, row| {
                for f in 0..next_freq {
                    let slot = &mut row[f * channels..(f + 1) * channels];
                    for dt in 0..KERNEL {
                        for df in 0..KERNEL {
                            let (source_t, source_f) = ((t * STRIDE + dt) as isize - 1, (f * STRIDE + df) as isize - 1);
                            if source_t < 0 || source_f < 0 || source_t as usize >= time || source_f as usize >= freq {
                                continue;
                            }
                            let base = (source_t as usize * freq + source_f as usize) * channels;
                            let weights = &depthwise[(dt * KERNEL + df) * channels..(dt * KERNEL + df + 1) * channels];
                            for channel in 0..channels {
                                slot[channel] += current[base + channel] * weights[channel];
                            }
                        }
                    }
                }
            });
            mask_rows(&mut downsampled, next_time, next_freq * channels, length);
            let positions = next_time * next_freq;
            let mut rotated = Vec::new();
            pointwise.prepare_batch(&downsampled, positions, &mut rotated);
            let mut projected = vec![0.0f32; positions * channels];
            pointwise.multiply_batch(&rotated, positions, &mut projected);
            current = projected;
            time = next_time;
            freq = next_freq;
        }
        current.iter_mut().for_each(|value| *value = silu(*value));
        let width = channels * freq;
        let mut flattened = vec![0.0f32; time * width];
        for t in 0..time {
            for f in 0..freq {
                for c in 0..channels {
                    flattened[t * width + c * freq + f] = current[(t * freq + f) * channels + c];
                }
            }
        }
        flattened.chunks_exact_mut(width).for_each(fake_quantise_int8);
        let mut rotated = Vec::new();
        self.out.prepare_batch(&flattened, time, &mut rotated);
        let mut data = vec![0.0f32; time * self.out.rows()];
        self.out.multiply_batch(&rotated, time, &mut data);
        StemOutput { data, frames: time, valid: length }
    }
}

fn mask_rows(values: &mut [f32], rows: usize, row_width: usize, valid: usize) {
    for row in valid..rows {
        values[row * row_width..(row + 1) * row_width].iter_mut().for_each(|v| *v = 0.0);
    }
}
