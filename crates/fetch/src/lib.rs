use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const HUB: &str = "https://huggingface.co";
const NEEDLE_REPO: &str = "Cactus-Compute/needle3";
const WHISTLE_REPO: &str = "Cactus-Compute/whistle";
const CACHE_ENV: &str = "RUSTYNEEDLE_CACHE_DIR";
const OFFLINE_ENV: &str = "HF_HUB_OFFLINE";

#[derive(Debug)]
pub struct FetchError(pub String);

impl fmt::Display for FetchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for FetchError {}

impl From<io::Error> for FetchError {
    fn from(error: io::Error) -> Self {
        FetchError(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
    Needle,
    Whistle,
}

impl Base {
    pub fn repository(self) -> &'static str {
        match self {
            Base::Needle => NEEDLE_REPO,
            Base::Whistle => WHISTLE_REPO,
        }
    }

    pub fn file_name(self) -> &'static str {
        match self {
            Base::Needle => "needle3.cact",
            Base::Whistle => "whistle.cact",
        }
    }
}

pub fn cache_directory() -> PathBuf {
    if let Some(custom) = std::env::var_os(CACHE_ENV) {
        return PathBuf::from(custom);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("rustyneedle")
}

pub fn url(repository: &str, file: &str) -> String {
    format!("{HUB}/{repository}/resolve/main/{file}")
}

pub fn download(repository: &str, file: &str, destination: &Path) -> Result<PathBuf, FetchError> {
    if std::env::var_os(OFFLINE_ENV).is_some() {
        return Err(FetchError(format!("{file} is not on disk and {OFFLINE_ENV} forbids downloading it")));
    }
    let response = ureq::get(&url(repository, file)).call().map_err(|error| FetchError(format!("downloading {repository}/{file}: {error}")))?;
    fs::create_dir_all(destination.parent().unwrap_or_else(|| Path::new(".")))?;
    let partial = destination.with_extension("part");
    let mut reader = response.into_reader();
    let mut out = File::create(&partial)?;
    io::copy(&mut reader, &mut out)?;
    out.flush()?;
    fs::rename(&partial, destination)?;
    Ok(destination.to_path_buf())
}

pub fn base_weights(base: Base) -> Result<PathBuf, FetchError> {
    let cached = cache_directory().join(base.file_name());
    if cached.exists() {
        return Ok(cached);
    }
    download(base.repository(), base.file_name(), &cached)
}

pub enum Spec {
    Base(Base),
    Hub { repository: String, file: String },
}

pub fn parse_spec(spec: &str) -> Result<Spec, FetchError> {
    match spec {
        "needle3" | "needle" => Ok(Spec::Base(Base::Needle)),
        "whistle" => Ok(Spec::Base(Base::Whistle)),
        other => {
            let parts: Vec<&str> = other.splitn(3, '/').collect();
            match parts.as_slice() {
                [org, repo] => Ok(Spec::Hub { repository: format!("{org}/{repo}"), file: format!("{repo}.cact") }),
                [org, repo, file] => Ok(Spec::Hub { repository: format!("{org}/{repo}"), file: (*file).to_owned() }),
                _ => Err(FetchError(format!("unrecognised download spec {other:?}; use needle3, whistle or <org>/<repo>[/<file>.cact]"))),
            }
        }
    }
}

pub fn fetch_spec(spec: &str, directory: &Path) -> Result<PathBuf, FetchError> {
    let (repository, file) = match parse_spec(spec)? {
        Spec::Base(base) => (base.repository().to_owned(), base.file_name().to_owned()),
        Spec::Hub { repository, file } => (repository, file),
    };
    download(&repository, &file, &directory.join(&file))
}
