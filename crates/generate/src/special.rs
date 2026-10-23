use tokenizer::Tokenizer;

#[derive(Clone, Copy, Debug)]
pub struct SpecialTokens {
    pub bos: u32,
    pub eos: u32,
    pub im_end: u32,
    pub think_start: u32,
    pub think_end: u32,
    pub tool_call_start: u32,
    pub tool_call_end: u32,
}

impl SpecialTokens {
    pub fn resolve(tokenizer: &Tokenizer) -> Self {
        let vocabulary = tokenizer.vocabulary();
        let id = |piece: &str| vocabulary.piece_id(piece).unwrap_or_else(|| panic!("vocabulary lacks {piece}"));
        Self {
            bos: vocabulary.bos_id(),
            eos: vocabulary.eos_id(),
            im_end: id("<|im_end|>"),
            think_start: id("<think>"),
            think_end: id("</think>"),
            tool_call_start: id("<tool_call>"),
            tool_call_end: id("</tool_call>"),
        }
    }
}
