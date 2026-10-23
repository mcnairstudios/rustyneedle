use rustyneedle_cact::{Archive, ElementKind, Layout};

fn whistle_path() -> Option<std::path::PathBuf> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/whistle.cact");
    path.exists().then_some(path)
}

#[test]
fn whistle_geometry_matches_the_header() {
    let Some(path) = whistle_path() else { return };
    let archive = Archive::open(path).unwrap();
    let geometry = archive.geometry();
    assert_eq!(geometry.vocab, 8199);
    assert_eq!(geometry.d_model, 512);
    assert_eq!(geometry.num_layers, 8);
    assert_eq!(geometry.engram_orders, vec![2, 3]);
    assert_eq!(geometry.engram_sites, vec![3, 7]);
    assert_eq!(geometry.codebook_len, 28);
    assert_eq!(archive.records().len(), 681);
}

#[test]
fn whistle_layout_consumes_every_tensor() {
    let Some(path) = whistle_path() else { return };
    let archive = Archive::open(path).unwrap();
    let layout = Layout::resolve(&archive).unwrap();
    let audio = layout.audio.as_ref().expect("whistle carries an audio encoder");
    assert_eq!(audio.manifest.encoder_layers, 8);
    assert_eq!(audio.manifest.mels, 80);
    assert_eq!(audio.manifest.hop, 160);
    assert_eq!(layout.layers.len(), 8);
    assert_eq!(layout.engrams.len(), 2);
    assert!(layout.heads.is_empty());
    assert_eq!(archive.record(layout.tokenizer).kind, ElementKind::Raw);
}

#[test]
fn needle_layout_consumes_every_tensor() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/needle3.cact");
    if !path.exists() {
        return;
    }
    let archive = Archive::open(path).unwrap();
    let layout = Layout::resolve(&archive).unwrap();
    let geometry = archive.geometry();
    println!("{geometry:#?} heads={}", layout.heads.len());
    assert_eq!(layout.layers.len(), geometry.num_layers);
    assert!(layout.audio.is_none());
}
