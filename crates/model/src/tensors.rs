use cact::{half_to_f32, Archive, ElementKind};

pub fn floats(archive: &Archive, index: usize) -> Vec<f32> {
    match archive.record(index).kind {
        ElementKind::Half => archive.halves(index).iter().map(|&bits| half_to_f32(bits)).collect(),
        ElementKind::Single => archive.singles(index).to_vec(),
        other => panic!("tensor {index} holds {other:?}, not floating point values"),
    }
}

pub fn scalar(archive: &Archive, index: usize) -> f32 {
    floats(archive, index)[0]
}
