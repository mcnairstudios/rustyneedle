use crate::cursor::Cursor;
use crate::error::{CactError, Result};

pub const FORMAT_TAG: u32 = 0x05E1_2A84;
pub const HEADER_BYTES: usize = 196;
pub const MAX_ENGRAM_ORDERS: usize = 4;
pub const MAX_ENGRAM_SITES: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    pub tensor_count: usize,
    pub codebook_len: usize,
    pub kv_window: usize,
    pub kv_bits: u32,
    pub vocab: usize,
    pub out_vocab: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub max_seq_len: usize,
    pub hada_n: usize,
    pub mhc_lanes: usize,
    pub sliding_window: usize,
    pub global_layers: u64,
    pub qkv_conv_taps: usize,
    pub engram_slots: usize,
    pub engram_sub_dim: usize,
    pub engram_tables: usize,
    pub engram_conv_taps: usize,
    pub engram_conv_dilation: usize,
    pub engram_seed_heads: usize,
    pub engram_orders: Vec<usize>,
    pub engram_sites: Vec<usize>,
    pub rope_theta: f32,
}

impl Geometry {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes, 0, "header");
        let tag = cursor.u32()?;
        if tag != FORMAT_TAG {
            return Err(CactError::WrongTag { found: tag });
        }
        let tensor_count = cursor.u32()? as usize;
        let codebook_len = cursor.u32()? as usize;
        let kv_window = cursor.u32()? as usize;
        let kv_bits = cursor.u32()?;
        let vocab = cursor.u32()? as usize;
        let out_vocab = cursor.u32()? as usize;
        let d_model = cursor.u32()? as usize;
        let num_heads = cursor.u32()? as usize;
        let num_kv_heads = cursor.u32()? as usize;
        let num_layers = cursor.u32()? as usize;
        let qk_head_dim = cursor.u32()? as usize;
        let v_head_dim = cursor.u32()? as usize;
        let max_seq_len = cursor.u32()? as usize;
        let hada_n = cursor.u32()? as usize;
        let mhc_lanes = cursor.u32()? as usize;
        let sliding_window = cursor.u32()? as usize;
        let global_low = cursor.u32()? as u64;
        let global_high = cursor.u32()? as u64;
        let qkv_conv_taps = cursor.u32()? as usize;
        let engram_slots = cursor.u32()? as usize;
        let engram_sub_dim = cursor.u32()? as usize;
        let engram_tables = cursor.u32()? as usize;
        let engram_conv_taps = cursor.u32()? as usize;
        let engram_conv_dilation = cursor.u32()? as usize;
        let engram_seed_heads = cursor.u32()? as usize;
        let order_count = cursor.u32()? as usize;
        let mut orders = [0usize; MAX_ENGRAM_ORDERS];
        for slot in orders.iter_mut() {
            *slot = cursor.u32()? as usize;
        }
        let site_count = cursor.u32()? as usize;
        let mut sites = [0usize; MAX_ENGRAM_SITES];
        for slot in sites.iter_mut() {
            *slot = cursor.u32()? as usize;
        }
        let rope_theta = cursor.f32()?;
        if order_count > MAX_ENGRAM_ORDERS || site_count > MAX_ENGRAM_SITES {
            return Err(CactError::BadLayout { reason: "engram order or site count exceeds the header capacity".into() });
        }
        Ok(Self {
            tensor_count,
            codebook_len,
            kv_window,
            kv_bits,
            vocab,
            out_vocab,
            d_model,
            num_heads,
            num_kv_heads,
            num_layers,
            qk_head_dim,
            v_head_dim,
            max_seq_len,
            hada_n,
            mhc_lanes,
            sliding_window,
            global_layers: global_low | (global_high << 32),
            qkv_conv_taps,
            engram_slots,
            engram_sub_dim,
            engram_tables,
            engram_conv_taps,
            engram_conv_dilation,
            engram_seed_heads,
            engram_orders: orders[..order_count].to_vec(),
            engram_sites: sites[..site_count].to_vec(),
            rope_theta,
        })
    }

    pub fn is_global_layer(&self, layer: usize) -> bool {
        self.global_layers >> layer & 1 == 1
    }

    pub fn output_vocab(&self) -> usize {
        if self.out_vocab == 0 {
            self.vocab
        } else {
            self.out_vocab
        }
    }

    pub fn query_width(&self) -> usize {
        self.num_heads * self.qk_head_dim
    }

    pub fn key_width(&self) -> usize {
        self.num_kv_heads * self.qk_head_dim
    }

    pub fn value_width(&self) -> usize {
        self.num_kv_heads * self.v_head_dim
    }

    pub fn attention_width(&self) -> usize {
        self.num_heads * self.v_head_dim
    }
}
