use crate::archive::Archive;
use crate::error::{CactError, Result};
use crate::record::ElementKind;

pub const AUDIO_MANIFEST_MAGIC: f32 = 3246.0;
const AUDIO_MANIFEST_LEN: usize = 13;
const HEAD_MANIFEST_CODES: [u8; 3] = [1, 2, 3];

struct Sequence<'a> {
    archive: &'a Archive,
    next: usize,
    end: usize,
}

impl<'a> Sequence<'a> {
    fn take(&mut self) -> Result<usize> {
        if self.next >= self.end {
            return Err(CactError::BadLayout { reason: "directory ends before the canonical tensor list".into() });
        }
        let index = self.next;
        self.next += 1;
        Ok(index)
    }

    fn take_array<const N: usize>(&mut self) -> Result<[usize; N]> {
        let mut out = [0usize; N];
        for slot in out.iter_mut() {
            *slot = self.take()?;
        }
        Ok(out)
    }

    fn peek_kind(&self) -> Option<ElementKind> {
        (self.next < self.end).then(|| self.archive.record(self.next).kind)
    }
}

#[derive(Clone, Debug)]
pub struct LayerTensors {
    pub norm_in: usize,
    pub q_proj: usize,
    pub k_proj: usize,
    pub v_proj: usize,
    pub conv_taps: Option<[usize; 3]>,
    pub q_norm: usize,
    pub k_norm: usize,
    pub gate_proj: usize,
    pub out_proj: usize,
    pub post_norm: usize,
    pub attn_gate: usize,
    pub pre_hada: usize,
    pub hadamard: HadamardTensors,
}

#[derive(Clone, Debug)]
pub struct HadamardTensors {
    pub d1: usize,
    pub d2: usize,
    pub b2: usize,
    pub d3: usize,
    pub d4: usize,
    pub w1a: usize,
    pub w1b: usize,
    pub w2a: usize,
    pub w2b: usize,
    pub w3a: usize,
    pub w3b: usize,
    pub cond_v: usize,
    pub cond_u: usize,
}

#[derive(Clone, Debug)]
pub struct LaneTensors {
    pub a_pre: usize,
    pub a_post: usize,
    pub a_res: usize,
    pub b_pre: usize,
    pub b_post: usize,
    pub b_res: usize,
    pub phi_pre: usize,
    pub phi_post: usize,
    pub phi_res: usize,
}

#[derive(Clone, Debug)]
pub struct EngramTensors {
    pub tables: usize,
    pub key_proj: usize,
    pub value_proj: usize,
    pub taps: usize,
}

#[derive(Clone, Debug)]
pub struct HeadTensors {
    pub code: u8,
    pub probes: usize,
    pub gain: usize,
    pub query: usize,
    pub row_bias: usize,
    pub proj: usize,
    pub bias: usize,
    pub calibration: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct AudioManifest {
    pub text_vocab: usize,
    pub encoder_layers: usize,
    pub encoder_lanes: usize,
    pub conv_kernel: usize,
    pub stem_channels: usize,
    pub stem_stages: usize,
    pub mels: usize,
    pub sample_rate: usize,
    pub fft_size: usize,
    pub window: usize,
    pub hop: usize,
}

#[derive(Clone, Debug)]
pub struct CrossTensors {
    pub norm: usize,
    pub q_proj: usize,
    pub k_proj: usize,
    pub v_proj: usize,
    pub gate_proj: usize,
    pub out_proj: usize,
    pub q_norm: usize,
    pub k_norm: usize,
    pub post_norm: usize,
    pub gate: usize,
}

#[derive(Clone, Debug)]
pub struct EncoderTensors {
    pub pre_hada_norm_first: usize,
    pub attn_norm: usize,
    pub post_attn_norm: usize,
    pub conv_norm: usize,
    pub conv_out_norm: usize,
    pub pre_hada_norm: usize,
    pub q_proj: usize,
    pub k_proj: usize,
    pub v_proj: usize,
    pub gate_proj: usize,
    pub out_proj: usize,
    pub q_norm: usize,
    pub k_norm: usize,
    pub attn_gate: usize,
    pub pw1: usize,
    pub pw2: usize,
    pub depthwise: usize,
    pub hadamard_first: HadamardTensors,
    pub hadamard: HadamardTensors,
}

#[derive(Clone, Debug)]
pub struct StemTensors {
    pub first: usize,
    pub stages: Vec<(usize, usize)>,
    pub out: usize,
}

#[derive(Clone, Debug)]
pub struct AudioLayout {
    pub manifest: AudioManifest,
    pub cross: Vec<CrossTensors>,
    pub position_gate: usize,
    pub encoder: Vec<EncoderTensors>,
    pub encoder_lanes: LaneTensors,
    pub encoder_final_norm: usize,
    pub stem: StemTensors,
    pub mel: usize,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub embedding: usize,
    pub layers: Vec<LayerTensors>,
    pub lanes: LaneTensors,
    pub hadamard_perms: [usize; 2],
    pub engrams: Vec<EngramTensors>,
    pub final_norm: usize,
    pub heads: Vec<HeadTensors>,
    pub audio: Option<AudioLayout>,
    pub tokenizer: usize,
}

impl Layout {
    pub fn resolve(archive: &Archive) -> Result<Self> {
        let geometry = archive.geometry();
        let count = archive.records().len();
        let tokenizer = count.checked_sub(1).filter(|&last| archive.record(last).kind == ElementKind::Raw);
        let tokenizer = tokenizer.ok_or_else(|| CactError::BadLayout { reason: "no trailing tokenizer blob".into() })?;
        let mut sequence = Sequence { archive, next: 0, end: tokenizer };

        let embedding = sequence.take()?;
        let layers = (0..geometry.num_layers).map(|_| read_layer(&mut sequence, geometry.qkv_conv_taps > 0)).collect::<Result<Vec<_>>>()?;
        let lanes = read_lanes(&mut sequence)?;
        let hadamard_perms = sequence.take_array::<2>()?;
        let engrams = (0..geometry.engram_sites.len()).map(|_| read_engram(&mut sequence)).collect::<Result<Vec<_>>>()?;
        let final_norm = sequence.take()?;
        let heads = read_heads(&mut sequence)?;
        let audio = read_audio(&mut sequence, geometry.num_layers)?;
        if sequence.next != tokenizer {
            return Err(CactError::BadLayout { reason: format!("{} tensors are left over before the tokenizer", tokenizer - sequence.next) });
        }
        Ok(Self { embedding, layers, lanes, hadamard_perms, engrams, final_norm, heads, audio, tokenizer })
    }
}

fn read_hadamard(sequence: &mut Sequence<'_>) -> Result<HadamardTensors> {
    let [d1, d2, b2, d3, d4, w1a, w1b, w2a, w2b, w3a, w3b, cond_v, cond_u] = sequence.take_array::<13>()?;
    Ok(HadamardTensors { d1, d2, b2, d3, d4, w1a, w1b, w2a, w2b, w3a, w3b, cond_v, cond_u })
}

fn read_layer(sequence: &mut Sequence<'_>, with_taps: bool) -> Result<LayerTensors> {
    let norm_in = sequence.take()?;
    let [q_proj, k_proj, v_proj] = sequence.take_array::<3>()?;
    let conv_taps = if with_taps { Some(sequence.take_array::<3>()?) } else { None };
    let [q_norm, k_norm, gate_proj, out_proj, post_norm, attn_gate, pre_hada] = sequence.take_array::<7>()?;
    let hadamard = read_hadamard(sequence)?;
    Ok(LayerTensors { norm_in, q_proj, k_proj, v_proj, conv_taps, q_norm, k_norm, gate_proj, out_proj, post_norm, attn_gate, pre_hada, hadamard })
}

fn read_lanes(sequence: &mut Sequence<'_>) -> Result<LaneTensors> {
    let [a_pre, a_post, a_res, b_pre, b_post, b_res, phi_pre, phi_post, phi_res] = sequence.take_array::<9>()?;
    Ok(LaneTensors { a_pre, a_post, a_res, b_pre, b_post, b_res, phi_pre, phi_post, phi_res })
}

fn read_engram(sequence: &mut Sequence<'_>) -> Result<EngramTensors> {
    let [tables, key_proj, value_proj, taps] = sequence.take_array::<4>()?;
    Ok(EngramTensors { tables, key_proj, value_proj, taps })
}

fn read_heads(sequence: &mut Sequence<'_>) -> Result<Vec<HeadTensors>> {
    let manifest_follows = sequence.peek_kind() == Some(ElementKind::Half);
    if !manifest_follows {
        return Ok(Vec::new());
    }
    let manifest = sequence.take()?;
    let codes: Vec<u8> = sequence.archive.halves(manifest).iter().map(|&bits| half_to_f32(bits).round() as u8).collect();
    if codes.iter().any(|code| !HEAD_MANIFEST_CODES.contains(code)) {
        return Err(CactError::BadLayout { reason: "unrecognised head manifest code".into() });
    }
    codes
        .into_iter()
        .map(|code| {
            let [probes, gain, query, row_bias, proj, bias] = sequence.take_array::<6>()?;
            let calibration = if code == 3 { Some(sequence.take()?) } else { None };
            Ok(HeadTensors { code, probes, gain, query, row_bias, proj, bias, calibration })
        })
        .collect()
}

fn read_audio(sequence: &mut Sequence<'_>, decoder_layers: usize) -> Result<Option<AudioLayout>> {
    if sequence.peek_kind() != Some(ElementKind::Single) {
        return Ok(None);
    }
    let manifest_index = sequence.take()?;
    let values = sequence.archive.singles(manifest_index);
    if values.len() != AUDIO_MANIFEST_LEN || values[0] != AUDIO_MANIFEST_MAGIC {
        return Err(CactError::BadLayout { reason: "audio manifest is malformed".into() });
    }
    let manifest = AudioManifest {
        text_vocab: values[2] as usize,
        encoder_layers: values[3] as usize,
        encoder_lanes: values[4] as usize,
        conv_kernel: values[5] as usize,
        stem_channels: values[6] as usize,
        stem_stages: values[7] as usize,
        mels: values[8] as usize,
        sample_rate: values[9] as usize,
        fft_size: values[10] as usize,
        window: values[11] as usize,
        hop: values[12] as usize,
    };
    let cross = (0..decoder_layers)
        .map(|_| {
            let [norm, q_proj, k_proj, v_proj, gate_proj, out_proj, q_norm, k_norm, post_norm, gate] = sequence.take_array::<10>()?;
            Ok(CrossTensors { norm, q_proj, k_proj, v_proj, gate_proj, out_proj, q_norm, k_norm, post_norm, gate })
        })
        .collect::<Result<Vec<_>>>()?;
    let position_gate = sequence.take()?;
    let encoder = (0..manifest.encoder_layers).map(|_| read_encoder(sequence)).collect::<Result<Vec<_>>>()?;
    let encoder_lanes = read_lanes(sequence)?;
    let encoder_final_norm = sequence.take()?;
    let first = sequence.take()?;
    let stages = (1..manifest.stem_stages).map(|_| Ok((sequence.take()?, sequence.take()?))).collect::<Result<Vec<_>>>()?;
    let out = sequence.take()?;
    let mel = sequence.take()?;
    Ok(Some(AudioLayout { manifest, cross, position_gate, encoder, encoder_lanes, encoder_final_norm, stem: StemTensors { first, stages, out }, mel }))
}

fn read_encoder(sequence: &mut Sequence<'_>) -> Result<EncoderTensors> {
    let [pre_hada_norm_first, attn_norm, post_attn_norm, conv_norm, conv_out_norm, pre_hada_norm] = sequence.take_array::<6>()?;
    let [q_proj, k_proj, v_proj, gate_proj, out_proj, q_norm, k_norm, attn_gate, pw1, pw2, depthwise] = sequence.take_array::<11>()?;
    let hadamard_first = read_hadamard(sequence)?;
    let hadamard = read_hadamard(sequence)?;
    Ok(EncoderTensors {
        pre_hada_norm_first,
        attn_norm,
        post_attn_norm,
        conv_norm,
        conv_out_norm,
        pre_hada_norm,
        q_proj,
        k_proj,
        v_proj,
        gate_proj,
        out_proj,
        q_norm,
        k_norm,
        attn_gate,
        pw1,
        pw2,
        depthwise,
        hadamard_first,
        hadamard,
    })
}

pub fn half_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exponent = (bits >> 10 & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let value = match (exponent, mantissa) {
        (0, 0) => sign << 31,
        (0, _) => {
            let shift = mantissa.leading_zeros() - 21;
            let mantissa = (mantissa << shift) & 0x3ff;
            sign << 31 | (113 - shift) << 23 | mantissa << 13
        }
        (0x1f, 0) => sign << 31 | 0x7f80_0000,
        (0x1f, _) => sign << 31 | 0x7fc0_0000 | mantissa << 13,
        _ => sign << 31 | (exponent + 112) << 23 | mantissa << 13,
    };
    f32::from_bits(value)
}
