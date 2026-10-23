use rustyneedle_fetch::{parse_spec, url, Base, Spec};

#[test]
fn named_bases_resolve_to_their_repositories() {
    match parse_spec("needle3").unwrap() {
        Spec::Base(base) => assert_eq!(base, Base::Needle),
        _ => panic!("expected a base"),
    }
    assert_eq!(Base::Whistle.repository(), "Cactus-Compute/whistle");
    assert_eq!(Base::Needle.file_name(), "needle3.cact");
}

#[test]
fn hub_specs_default_the_file_to_the_repository_name() {
    match parse_spec("acme/widget").unwrap() {
        Spec::Hub { repository, file } => assert_eq!((repository.as_str(), file.as_str()), ("acme/widget", "widget.cact")),
        _ => panic!("expected a hub spec"),
    }
    match parse_spec("acme/widget/tuned.cact").unwrap() {
        Spec::Hub { file, .. } => assert_eq!(file, "tuned.cact"),
        _ => panic!("expected a hub spec"),
    }
    assert!(parse_spec("nonsense").is_err());
}

#[test]
fn urls_point_at_the_hub_resolve_endpoint() {
    assert_eq!(url("Cactus-Compute/whistle", "whistle.cact"), "https://huggingface.co/Cactus-Compute/whistle/resolve/main/whistle.cact");
}

#[test]
fn offline_mode_refuses_to_download() {
    std::env::set_var("HF_HUB_OFFLINE", "1");
    let target = std::env::temp_dir().join("needle-fetch-offline-test.cact");
    assert!(rustyneedle_fetch::download("Cactus-Compute/whistle", "whistle.cact", &target).is_err());
    std::env::remove_var("HF_HUB_OFFLINE");
}
