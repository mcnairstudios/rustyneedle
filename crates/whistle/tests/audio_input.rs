use rustyneedle_whistle::{decode_wav, resample, Options, Whistle};

fn wav(format_tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8], extra_chunk: bool) -> Vec<u8> {
    let align = channels * bits / 8;
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&format_tag.to_le_bytes());
    fmt.extend_from_slice(&channels.to_le_bytes());
    fmt.extend_from_slice(&rate.to_le_bytes());
    fmt.extend_from_slice(&(rate * align as u32).to_le_bytes());
    fmt.extend_from_slice(&align.to_le_bytes());
    fmt.extend_from_slice(&bits.to_le_bytes());
    let mut body = Vec::new();
    body.extend_from_slice(b"WAVE");
    if extra_chunk {
        body.extend_from_slice(b"LIST");
        body.extend_from_slice(&3u32.to_le_bytes());
        body.extend_from_slice(&[1, 2, 3, 0]);
    }
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    body.extend_from_slice(&fmt);
    body.extend_from_slice(b"data");
    body.extend_from_slice(&(data.len() as u32).to_le_bytes());
    body.extend_from_slice(data);
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

#[test]
fn sixteen_bit_pcm_is_scaled_to_unit_range() {
    let data: Vec<u8> = [0i16, 16384, -16384, 32767, -32768].iter().flat_map(|v| v.to_le_bytes()).collect();
    let samples = decode_wav(&wav(1, 1, 16000, 16, &data, false)).unwrap();
    assert_eq!(samples.len(), 5);
    assert!((samples[1] - 0.5).abs() < 1e-4 && (samples[2] + 0.5).abs() < 1e-4 && samples[4] == -1.0);
}

#[test]
fn eight_bit_pcm_is_unsigned_and_centred() {
    let samples = decode_wav(&wav(1, 1, 16000, 8, &[128, 255, 0], false)).unwrap();
    assert!(samples[0].abs() < 1e-6 && samples[1] > 0.99 && samples[2] == -1.0);
}

#[test]
fn twenty_four_bit_pcm_keeps_its_sign() {
    let positive = (0x400000i32).to_le_bytes();
    let negative = (-0x400000i32).to_le_bytes();
    let data = [&positive[..3], &negative[..3]].concat();
    let samples = decode_wav(&wav(1, 1, 16000, 24, &data, false)).unwrap();
    assert!((samples[0] - 0.5).abs() < 1e-4 && (samples[1] + 0.5).abs() < 1e-4);
}

#[test]
fn float_wav_passes_through_and_stereo_is_averaged() {
    let data: Vec<u8> = [0.25f32, 0.75, -0.5, 0.5].iter().flat_map(|v| v.to_le_bytes()).collect();
    let samples = decode_wav(&wav(3, 2, 16000, 32, &data, false)).unwrap();
    assert_eq!(samples, vec![0.5, 0.0]);
}

#[test]
fn unknown_chunks_before_the_format_are_skipped() {
    let data: Vec<u8> = [100i16, 200].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(decode_wav(&wav(1, 1, 16000, 16, &data, true)).unwrap().len(), 2);
}

#[test]
fn files_that_are_not_wav_are_rejected_with_a_message() {
    assert!(decode_wav(b"not a wav file at all").is_err());
    assert!(decode_wav(&wav(2, 1, 16000, 16, &[0, 0], false)).is_err());
}

#[test]
fn resampling_changes_the_length_and_keeps_the_tone() {
    let rate = 44100.0f32;
    let tone: Vec<f32> = (0..44100).map(|n| (2.0 * std::f32::consts::PI * 440.0 * n as f32 / rate).sin() * 0.5).collect();
    let converted = resample(&tone, 44100, 16000);
    assert!((converted.len() as i64 - 16000).abs() <= 1);
    let crossings = converted.windows(2).filter(|pair| pair[0] <= 0.0 && pair[1] > 0.0).count();
    assert!((430..=450).contains(&crossings), "{crossings} rising crossings in one second of 440 Hz");
    let peak = converted[1000..15000].iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    assert!((peak - 0.5).abs() < 0.05, "{peak}");
    assert_eq!(resample(&tone, 44100, 44100), tone);
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn degenerate_audio_never_panics() {
    let whistle = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap();
    let options = Options { word_timestamps: true, ..Options::default() };
    let mut state = 7u32;
    let mut noise = |count: usize, level: f32| -> Vec<f32> {
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 8) as f32 / 8388608.0 - 1.0) * level
            })
            .collect()
    };
    for samples in [Vec::new(), vec![0.3], noise(100, 0.5), noise(1600, 0.5), noise(3200, 1.0), noise(16000, 1.0), vec![1.0; 16000], vec![-1.0; 8000]] {
        let transcript = whistle.transcribe(&samples, &options).unwrap();
        assert!(transcript.words.is_some());
    }
}

#[test]
fn keywords_can_be_given_as_one_newline_separated_text() {
    let options = Options::default().keywords_from_text("Maya Chen\n  living room \n\nKitchen\n");
    assert_eq!(options.keywords, vec!["Maya Chen", "living room", "Kitchen"]);
}

#[test]
fn raw_float_bytes_become_samples() {
    let bytes: Vec<u8> = [0.5f32, -0.25].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(rustyneedle_whistle::samples_from_bytes(&bytes).unwrap(), vec![0.5, -0.25]);
    assert!(rustyneedle_whistle::samples_from_bytes(&bytes[..7]).is_err());
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn missing_or_foreign_weights_fail_with_a_clear_message() {
    let missing = Whistle::open("/nonexistent/whistle.cact").err().expect("a missing file is an error");
    assert!(!missing.to_string().is_empty());
    let needle_only = Whistle::open(testkit::workspace_file("weights/needle3.cact")).err().expect("an archive without an audio encoder is refused");
    assert!(needle_only.to_string().contains("audio encoder"), "{needle_only}");
    let garbage = std::env::temp_dir().join("not-a-cact.bin");
    std::fs::write(&garbage, vec![7u8; 4096]).unwrap();
    assert!(Whistle::open(&garbage).err().expect("garbage is refused").to_string().contains("not a Needle 3 archive"));
}
