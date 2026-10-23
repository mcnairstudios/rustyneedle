use std::fmt;
use std::path::Path;

const TARGET_RATE: u32 = 16000;

#[derive(Debug)]
pub struct WavError(pub String);

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for WavError {}

struct Format {
    tag: u16,
    channels: usize,
    rate: u32,
    width: usize,
}

fn chunk<'a>(data: &'a [u8], name: &[u8; 4]) -> Option<&'a [u8]> {
    let mut at = 12;
    while at + 8 <= data.len() {
        let size = u32::from_le_bytes(data[at + 4..at + 8].try_into().ok()?) as usize;
        let end = (at + 8 + size).min(data.len());
        if &data[at..at + 4] == name {
            return Some(&data[at + 8..end]);
        }
        at += 8 + size + (size & 1);
    }
    None
}

pub fn read_wav(path: impl AsRef<Path>) -> Result<Vec<f32>, WavError> {
    let data = std::fs::read(path.as_ref()).map_err(|error| WavError(error.to_string()))?;
    decode_wav(&data)
}

pub fn decode_wav(data: &[u8]) -> Result<Vec<f32>, WavError> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(WavError("not a WAV file".into()));
    }
    let header = chunk(data, b"fmt ").ok_or_else(|| WavError("WAV has no fmt chunk".into()))?;
    let samples = chunk(data, b"data").ok_or_else(|| WavError("WAV has no data chunk".into()))?;
    if header.len() < 16 {
        return Err(WavError("WAV fmt chunk is truncated".into()));
    }
    let mut tag = u16::from_le_bytes([header[0], header[1]]);
    let channels = u16::from_le_bytes([header[2], header[3]]) as usize;
    let rate = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    let align = u16::from_le_bytes([header[12], header[13]]) as usize;
    if tag == 0xFFFE && header.len() >= 26 {
        tag = u16::from_le_bytes([header[24], header[25]]);
    }
    let format = Format { tag, channels: channels.max(1), rate, width: align / channels.max(1) };
    let mono = to_mono(&format, samples)?;
    Ok(if format.rate == TARGET_RATE { mono } else { resample(&mono, format.rate, TARGET_RATE) })
}

fn to_mono(format: &Format, raw: &[u8]) -> Result<Vec<f32>, WavError> {
    let frame_bytes = format.width * format.channels;
    let frames = raw.len() / frame_bytes.max(1);
    let mut out = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut total = 0.0f32;
        for channel in 0..format.channels {
            let at = frame * frame_bytes + channel * format.width;
            let bytes = &raw[at..at + format.width];
            total += match (format.tag, format.width) {
                (1, 1) => (bytes[0] as f32 - 128.0) / 128.0,
                (1, 2) => i16::from_le_bytes([bytes[0], bytes[1]]) as f32 / 32768.0,
                (1, 3) => (((bytes[0] as i32) | (bytes[1] as i32) << 8 | (bytes[2] as i32) << 16) << 8 >> 8) as f32 / 8_388_608.0,
                (1, 4) => i32::from_le_bytes(bytes.try_into().expect("four bytes")) as f32 / 2_147_483_648.0,
                (3, 4) => f32::from_le_bytes(bytes.try_into().expect("four bytes")),
                (3, 8) => f64::from_le_bytes(bytes.try_into().expect("eight bytes")) as f32,
                _ => return Err(WavError("not integer PCM or float WAV".into())),
            };
        }
        out.push(total / format.channels as f32);
    }
    Ok(out)
}

const SINC_TAPS: isize = 32;

pub fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if samples.is_empty() || from == to {
        return samples.to_vec();
    }
    let ratio = to as f64 / from as f64;
    let cutoff = ratio.min(1.0);
    let length = (samples.len() as f64 * ratio).floor() as usize;
    (0..length)
        .map(|index| {
            let centre = index as f64 / ratio;
            let first = centre.floor() as isize - SINC_TAPS + 1;
            let mut total = 0.0f64;
            for position in first..first + 2 * SINC_TAPS {
                if position < 0 || position as usize >= samples.len() {
                    continue;
                }
                let offset = centre - position as f64;
                let sinc = if offset.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * offset * cutoff).sin() / (std::f64::consts::PI * offset * cutoff) };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * offset / SINC_TAPS as f64).cos();
                total += samples[position as usize] as f64 * sinc * window * cutoff;
            }
            total as f32
        })
        .collect()
}
