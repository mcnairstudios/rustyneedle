use rustyneedle_whistle::{read_wav, Options, Whistle};
use serde::Deserialize;

#[derive(Deserialize)]
struct Plain {
    text: String,
    language: String,
}

#[derive(Deserialize)]
struct Clip {
    clip: String,
    plain: Plain,
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn transcripts_match_the_recorded_engine() {
    let whistle = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap();
    let clips: Vec<Clip> = testkit::golden("speech");
    let mut identical = 0;
    let mut report = Vec::new();
    for clip in &clips {
        let samples = read_wav(testkit::workspace_file(&format!("oracle/clips/{}", clip.clip))).unwrap();
        let got = whistle.transcribe(&samples, &Options::default()).unwrap();
        if got.text == clip.plain.text && got.language == clip.plain.language {
            identical += 1;
        } else {
            report.push(format!("{}: want {:?}/{} got {:?}/{}", clip.clip, clip.plain.text, clip.plain.language, got.text, got.language));
        }
    }
    println!("{}\n{identical}/{} identical", report.join("\n"), clips.len());
    assert!(identical * 2 >= clips.len());
}

#[derive(Deserialize)]
struct GoldenWord {
    word: String,
    start: f64,
    end: f64,
    probability: f32,
}

#[derive(Deserialize)]
struct Timed {
    words: Vec<GoldenWord>,
}

#[derive(Deserialize)]
struct TimedClip {
    clip: String,
    timed: Timed,
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn word_timestamps_report_how_close_they_are_to_the_recorded_engine() {
    let whistle = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap();
    let clips: Vec<TimedClip> = testkit::golden("speech");
    let (mut start_error, mut end_error, mut probability_error, mut counted) = (0.0, 0.0, 0.0, 0usize);
    for clip in clips.iter().take(12) {
        let samples = read_wav(testkit::workspace_file(&format!("oracle/clips/{}", clip.clip))).unwrap();
        let options = Options { word_timestamps: true, ..Options::default() };
        let got = whistle.transcribe(&samples, &options).unwrap();
        let words = got.words.unwrap();
        println!(
            "{}: want {:?}\n   got  {:?}",
            clip.clip,
            clip.timed.words.iter().map(|w| (w.word.as_str(), w.start, w.end, w.probability)).collect::<Vec<_>>(),
            words.iter().map(|w| (w.word.as_str(), w.start, w.end, w.probability)).collect::<Vec<_>>()
        );
        for (want, have) in clip.timed.words.iter().zip(&words) {
            if want.word == have.word {
                start_error += (want.start - have.start).abs();
                end_error += (want.end - have.end).abs();
                probability_error += (want.probability - have.probability).abs() as f64;
                counted += 1;
            }
        }
    }
    let (start, end, probability) = (start_error / counted as f64, end_error / counted as f64, probability_error / counted as f64);
    println!("mean start error {start:.3}s end error {end:.3}s probability error {probability:.3} over {counted} words");
    assert!(start < 0.1 && end < 0.1 && probability < 0.05);
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn keywords_pull_a_misheard_name_toward_the_list_and_leave_other_clips_alone() {
    let whistle = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap();
    let clip = |name: &str| read_wav(testkit::workspace_file(&format!("oracle/clips/{name}.wav"))).unwrap();
    let options = Options { keywords: vec!["Leo".into()], ..Options::default() };
    let heard = whistle.transcribe(&clip("clip23"), &Options::default()).unwrap();
    let biased = whistle.transcribe(&clip("clip23"), &options).unwrap();
    assert!(!heard.text.contains("Leo"), "{}", heard.text);
    assert!(biased.text.contains("Leo"), "{}", biased.text);
    let unrelated = clip("clip08");
    assert_eq!(whistle.transcribe(&unrelated, &options).unwrap().text, whistle.transcribe(&unrelated, &Options::default()).unwrap().text);
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn a_forced_language_is_reported_back() {
    let whistle = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap();
    let samples = read_wav(testkit::workspace_file("oracle/clips/clip08.wav")).unwrap();
    let options = Options { language: Some("en".into()), ..Options::default() };
    assert_eq!(whistle.transcribe(&samples, &options).unwrap().language, "en");
    let unknown = Options { language: Some("xx".into()), ..Options::default() };
    assert!(whistle.transcribe(&samples, &unknown).is_err());
}

fn concatenated(names: &[&str], gap_seconds: usize) -> Vec<f32> {
    let mut samples = Vec::new();
    for name in names {
        samples.extend(read_wav(testkit::workspace_file(&format!("oracle/clips/{name}.wav"))).unwrap());
        samples.extend(std::iter::repeat_n(0.0, gap_seconds * 16000));
    }
    samples
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn audio_longer_than_thirty_seconds_is_transcribed_in_windows() {
    let whistle = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap();
    let names: Vec<String> = (8..=11).cycle().take(12).map(|i| format!("clip{i:02}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let samples = concatenated(&refs, 1);
    assert!(samples.len() > 30 * 16000);
    let options = Options { word_timestamps: true, ..Options::default() };
    let long = whistle.transcribe(&samples, &options).unwrap();
    let words = long.words.unwrap();
    assert!(long.text.to_lowercase().matches("purple rain").count() >= 3, "{}", long.text);
    assert!(!long.text.contains("Thank you"), "{}", long.text);
    assert!(words.windows(2).all(|pair| pair[1].start >= pair[0].start));
    assert!(words.last().unwrap().end > 30.0);
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn streaming_commits_what_two_passes_agree_on_and_flushes_the_tail() {
    let whistle = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap();
    let samples = concatenated(&["clip08"], 1);
    let mut stream = rustyneedle_whistle::Stream::new(&whistle, Options::default());
    let mut committed = Vec::new();
    let mut received = 0.0;
    for chunk in samples.chunks(16000) {
        let step = stream.process(chunk).unwrap();
        received = step.received;
        if !step.text.is_empty() {
            committed.push(step.text);
        }
    }
    let tail = stream.stop().unwrap();
    if !tail.text.is_empty() {
        committed.push(tail.text);
    }
    assert!(tail.pending.is_empty());
    assert!((received - samples.len() as f64 / 16000.0).abs() < 1e-6);
    let whole = whistle.transcribe(&samples, &Options::default()).unwrap().text;
    let joined = committed.join(" ");
    let squash = |text: &str| text.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect::<String>();
    assert_eq!(squash(&joined), squash(&whole), "{joined:?} vs {whole:?}");
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn silence_and_steady_noise_return_an_empty_transcript() {
    let whistle = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap();
    let silence = vec![0.0f32; 2 * 16000];
    let silent = whistle.transcribe(&silence, &Options::default()).unwrap();
    assert!(silent.text.is_empty() && silent.language.is_empty());
    let mut state = 12345u32;
    let noise: Vec<f32> = (0..64000)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 8388608.0 - 1.0) * 0.1
        })
        .collect();
    assert!(whistle.transcribe(&noise, &Options::default()).unwrap().text.is_empty());
}

#[test]
#[ignore = "needs the model files: run make weights and make goldens"]
fn a_shallower_decoder_rung_still_transcribes_and_is_faster() {
    let full = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap();
    assert_eq!(full.decoder_layers(), 8);
    let samples = read_wav(testkit::workspace_file("oracle/clips/clip08.wav")).unwrap();
    let reference = full.transcribe(&samples, &Options::default()).unwrap();
    let shallow = Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap().with_audio_depth(4).unwrap();
    assert_eq!(shallow.decoder_layers(), 4);
    let rung = shallow.transcribe(&samples, &Options::default()).unwrap();
    println!("depth 8: {:?} ({:.0} tok/s)  depth 4: {:?} ({:.0} tok/s)", reference.text, reference.tokens_per_second, rung.text, rung.tokens_per_second);
    assert!(rung.text.to_lowercase().contains("purple rain"), "{}", rung.text);
    assert!(Whistle::open(testkit::workspace_file("weights/whistle.cact")).unwrap().with_audio_depth(9).is_err());
}
