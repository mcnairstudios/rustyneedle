use std::sync::{Arc, OnceLock};

use kernels::{sigmoid, zero_centred_norm, zero_centred_norm_in_place};

use crate::engram::EngramOutput;
use crate::lanes::{LaneWeights, Routing};
use crate::session::{add_engram, lane_mean, Session};

impl Session {
    pub(crate) fn prefill(&mut self, new_tokens: &[u32]) {
        let model = Arc::clone(&self.model);
        let config = model.config();
        let weights = model.weights();
        let width = config.d_model;
        let lanes = config.lanes;
        let count = new_tokens.len();
        let first = self.tokens.len();
        self.tokens.extend_from_slice(new_tokens);
        let scale = (width as f32).sqrt();

        let frame = lanes * width;
        let mut stream = vec![0.0f32; count * frame];
        kernels::parallel_chunks_mut(&mut stream, frame, &|t, out| {
            let mut row = Vec::new();
            weights.embedding.dequantise_row(new_tokens[t] as usize, &mut row);
            for lane in out.chunks_exact_mut(width) {
                for (slot, value) in lane.iter_mut().zip(&row) {
                    *slot = value * scale;
                }
            }
        });
        let mut cells: Vec<Vec<Vec<f32>>> = stream.chunks_exact(frame).map(|lane_frame| vec![lane_mean(lane_frame, width, lanes)]).collect();

        let all_tokens = self.tokens.clone();
        let site_outputs: Vec<Option<Vec<EngramOutput>>> = self
            .engram_states
            .iter_mut()
            .zip(&weights.engrams)
            .zip(&config.engram.sites)
            .map(|((state, site_weights), site_layer)| {
                model.active_layers().contains(site_layer).then(|| state.contribute_batch(site_weights, &config.engram, &all_tokens, first, count))
            })
            .collect();

        let mut next_stream = vec![0.0f32; count * frame];
        let mut means = vec![0.0f32; count * width];
        for (slot, &layer) in model.active_layers().iter().enumerate() {
            let layer_weights = &weights.layers[layer];
            let engram_outputs = config.engram_site_at(layer).and_then(|site| site_outputs[site].as_ref());
            let routings: Vec<OnceLock<Routing>> = (0..count).map(|_| OnceLock::new()).collect();
            let mut x = vec![0.0f32; count * width];
            let mut normed = vec![0.0f32; count * width];
            kernels::parallel_chunks_pair_mut(&mut x, width, &mut normed, width, &|token, x_frame, normed_frame| {
                let routing = weights.lanes.route(layer, slot, &stream[token * frame..(token + 1) * frame], width);
                x_frame.copy_from_slice(&routing.input);
                if let Some(outputs) = engram_outputs {
                    add_engram(x_frame, &outputs[token], width);
                }
                zero_centred_norm(x_frame, &layer_weights.norm_in, normed_frame);
                routings[token].set(routing).ok().expect("each token is routed once");
            });
            let attended = layer_weights.attention.attend_batch(&mut self.caches[layer], config, layer, first, &normed, &self.frequencies);
            let gate = sigmoid(layer_weights.attention_gate);
            kernels::parallel_chunks_pair_mut(&mut next_stream, frame, &mut means, width, &|token, out, mean| {
                let routing = routings[token].get().expect("routed before the block");
                let mut block = x[token * width..(token + 1) * width].to_vec();
                let mut attention = attended[token * width..(token + 1) * width].to_vec();
                zero_centred_norm_in_place(&mut attention, &layer_weights.post_norm);
                for (slot, value) in block.iter_mut().zip(&attention) {
                    *slot += gate * value;
                }
                let mut pre = vec![0.0f32; width];
                zero_centred_norm(&block, &layer_weights.pre_hadamard_norm, &mut pre);
                let mlp = layer_weights.hadamard.apply(&pre);
                for (slot, value) in block.iter_mut().zip(&mlp) {
                    *slot += value;
                }
                let mut delta = vec![0.0f32; width];
                LaneWeights::block_delta(routing, &block, &mut delta);
                weights.lanes.combine_into(routing, &stream[token * frame..(token + 1) * frame], &delta, width, out);
                mean.copy_from_slice(&lane_mean(out, width, lanes));
            });
            std::mem::swap(&mut stream, &mut next_stream);
            for (token_cells, mean) in cells.iter_mut().zip(means.chunks_exact(width)) {
                token_cells.push(mean.to_vec());
            }
        }

        for token_cells in &cells {
            for (head, accumulator) in weights.heads.iter().zip(self.accumulators.iter_mut()) {
                head.observe(accumulator, token_cells);
            }
        }
        let last = lane_mean(&stream[(count - 1) * lanes * width..], width, lanes);
        let mut hidden = vec![0.0f32; width];
        zero_centred_norm(&last, &weights.final_norm, &mut hidden);
        self.hidden = hidden;
        self.cells = cells.pop().unwrap_or_default();
    }
}
