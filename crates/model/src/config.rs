use cact::Geometry;

#[derive(Clone, Debug)]
pub struct EngramConfig {
    pub slots: usize,
    pub sub_dim: usize,
    pub tables: usize,
    pub orders: Vec<usize>,
    pub heads_per_order: usize,
    pub conv_taps: usize,
    pub dilation: usize,
    pub seed_heads: usize,
    pub sites: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab: usize,
    pub output_vocab: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub qk_head_dim: usize,
    pub v_head_dim: usize,
    pub max_seq_len: usize,
    pub lanes: usize,
    pub sliding_window: usize,
    pub global_layers: u64,
    pub conv_taps: usize,
    pub rope_theta: f32,
    pub hadamard_size: usize,
    pub engram: EngramConfig,
}

impl ModelConfig {
    pub fn from_geometry(geometry: &Geometry) -> Self {
        let orders = geometry.engram_orders.clone();
        Self {
            vocab: geometry.vocab,
            output_vocab: geometry.output_vocab(),
            d_model: geometry.d_model,
            num_heads: geometry.num_heads,
            num_kv_heads: geometry.num_kv_heads,
            num_layers: geometry.num_layers,
            qk_head_dim: geometry.qk_head_dim,
            v_head_dim: geometry.v_head_dim,
            max_seq_len: geometry.max_seq_len,
            lanes: geometry.mhc_lanes,
            sliding_window: geometry.sliding_window,
            global_layers: geometry.global_layers,
            conv_taps: geometry.qkv_conv_taps,
            rope_theta: geometry.rope_theta,
            hadamard_size: geometry.hada_n,
            engram: EngramConfig {
                slots: geometry.engram_slots,
                sub_dim: geometry.engram_sub_dim,
                tables: geometry.engram_tables,
                heads_per_order: geometry.engram_tables / orders.len().max(1),
                orders,
                conv_taps: geometry.engram_conv_taps,
                dilation: geometry.engram_conv_dilation,
                seed_heads: geometry.engram_seed_heads,
                sites: geometry.engram_sites.clone(),
            },
        }
    }

    pub fn is_global_layer(&self, layer: usize) -> bool {
        self.global_layers >> layer & 1 == 1
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

    pub fn engram_site_at(&self, layer: usize) -> Option<usize> {
        self.engram.sites.iter().position(|&site| site == layer)
    }
}
