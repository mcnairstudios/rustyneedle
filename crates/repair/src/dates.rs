const MONTHS: &[(&str, u32)] = &[
    ("january", 1),
    ("february", 2),
    ("march", 3),
    ("april", 4),
    ("may", 5),
    ("june", 6),
    ("july", 7),
    ("august", 8),
    ("september", 9),
    ("october", 10),
    ("november", 11),
    ("december", 12),
    ("jan", 1),
    ("feb", 2),
    ("mar", 3),
    ("apr", 4),
    ("jun", 6),
    ("jul", 7),
    ("aug", 8),
    ("sep", 9),
    ("sept", 9),
    ("oct", 10),
    ("nov", 11),
    ("dec", 12),
];

fn month_number(word: &str) -> Option<u32> {
    let word = word.trim_end_matches('.').to_lowercase();
    MONTHS.iter().find(|(name, _)| *name == word).map(|(_, number)| *number)
}

fn day_number(word: &str) -> Option<u32> {
    let digits: String = word.chars().take_while(char::is_ascii_digit).collect();
    let suffix = &word[digits.len()..];
    if !matches!(suffix, "" | "st" | "nd" | "rd" | "th") {
        return None;
    }
    digits.parse().ok().filter(|day| (1..=31).contains(day))
}

fn year_number(word: &str) -> Option<u32> {
    (word.len() == 4 && word.chars().all(|c| c.is_ascii_digit())).then(|| word.parse().ok()).flatten()
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || c == ',').filter(|word| !word.is_empty()).map(str::to_owned).collect()
}

pub fn explicit_date(text: &str) -> Option<String> {
    let tokens = words(text);
    for token in &tokens {
        let parts: Vec<&str> = token.split('-').collect();
        if parts.len() == 3 && year_number(parts[0]).is_some() && parts[1].len() == 2 && parts[2].len() >= 2 {
            return Some(token.chars().take(10).collect());
        }
    }
    for window in tokens.windows(3) {
        if let (Some(day), Some(month), Some(year)) = (day_number(&window[0]), month_number(&window[1]), year_number(&window[2])) {
            return Some(format!("{year:04}-{month:02}-{day:02}"));
        }
        if let (Some(month), Some(day), Some(year)) = (month_number(&window[0]), day_number(&window[1]), year_number(&window[2])) {
            return Some(format!("{year:04}-{month:02}-{day:02}"));
        }
    }
    None
}

pub fn clock_times(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    let characters: Vec<char> = lowered.chars().collect();
    let mut times = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        if !characters[index].is_ascii_digit() || (index > 0 && (characters[index - 1].is_ascii_digit() || characters[index - 1] == ':')) {
            index += 1;
            continue;
        }
        let start = index;
        while index < characters.len() && characters[index].is_ascii_digit() {
            index += 1;
        }
        let hour_text: String = characters[start..index].iter().collect();
        let mut minute_text = String::from("00");
        let mut has_colon = false;
        if characters.get(index) == Some(&':') && characters.get(index + 1).is_some_and(char::is_ascii_digit) {
            has_colon = true;
            let minute_start = index + 1;
            let mut end = minute_start;
            while end < characters.len() && characters[end].is_ascii_digit() {
                end += 1;
            }
            minute_text = characters[minute_start..end].iter().collect();
            index = end;
        }
        let mut cursor = index;
        while characters.get(cursor) == Some(&' ') {
            cursor += 1;
        }
        let suffix: String = characters[cursor..].iter().take(2).collect();
        let meridiem = match suffix.as_str() {
            "am" => Some(false),
            "pm" => Some(true),
            _ => None,
        };
        if meridiem.is_none() && !has_colon {
            continue;
        }
        let Ok(mut hour) = hour_text.parse::<u32>() else { continue };
        if let Some(afternoon) = meridiem {
            hour %= 12;
            if afternoon {
                hour += 12;
            }
        }
        if hour < 24 && minute_text.len() == 2 {
            times.push(format!("{hour:02}:{minute_text}"));
        }
    }
    times
}
