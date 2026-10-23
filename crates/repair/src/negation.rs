const NEGATORS: &[&str] = &["don't", "dont", "do not", "never", "won't", "shouldn't", "no need to", "not to", "does not", "doesn't", "without"];

pub fn request_is_negated(query: &str) -> bool {
    let lowered = query.to_lowercase().replace('\u{2019}', "'");
    NEGATORS.iter().any(|negator| {
        lowered.match_indices(negator).any(|(at, _)| {
            let before_ok = at == 0 || !lowered[..at].chars().next_back().is_some_and(char::is_alphanumeric);
            let end = at + negator.len();
            let after_ok = !lowered[end..].chars().next().is_some_and(char::is_alphanumeric);
            before_ok && after_ok
        })
    })
}
