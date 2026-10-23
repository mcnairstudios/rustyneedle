use crate::dates::{clock_times, explicit_date};

const NUMBER_WORDS: &[(&str, f64)] = &[
    ("zero", 0.0),
    ("one", 1.0),
    ("two", 2.0),
    ("three", 3.0),
    ("four", 4.0),
    ("five", 5.0),
    ("six", 6.0),
    ("seven", 7.0),
    ("eight", 8.0),
    ("nine", 9.0),
    ("ten", 10.0),
    ("eleven", 11.0),
    ("twelve", 12.0),
    ("thirteen", 13.0),
    ("fourteen", 14.0),
    ("fifteen", 15.0),
    ("sixteen", 16.0),
    ("seventeen", 17.0),
    ("eighteen", 18.0),
    ("nineteen", 19.0),
    ("twenty", 20.0),
    ("thirty", 30.0),
    ("forty", 40.0),
    ("fifty", 50.0),
    ("sixty", 60.0),
    ("seventy", 70.0),
    ("eighty", 80.0),
    ("ninety", 90.0),
    ("hundred", 100.0),
    ("thousand", 1000.0),
    ("half", 0.5),
    ("dozen", 12.0),
];

pub struct Evidence {
    pub original: String,
    pub lowered: String,
    pub numbers: Vec<f64>,
    pub explicit_quantity: bool,
}

impl Evidence {
    pub fn new(text: &str) -> Self {
        let mut numbers = numbers_in(text);
        let explicit_quantity = !numbers.is_empty();
        if text.to_lowercase().split(|c: char| !c.is_alphabetic()).any(|word| word == "a" || word == "an") {
            numbers.push(1.0);
        }
        Evidence { original: text.to_owned(), lowered: text.to_lowercase(), numbers, explicit_quantity }
    }

    pub fn names_option(&self, option: &str) -> bool {
        let wanted = option.replace('_', " ").to_lowercase();
        let haystack = &self.lowered;
        haystack.match_indices(&wanted).any(|(at, _)| {
            let before = haystack[..at].chars().next_back().is_none_or(|c| !c.is_alphanumeric());
            let after = haystack[at + wanted.len()..].chars().next().is_none_or(|c| !c.is_alphanumeric());
            before && after
        })
    }

    pub fn mentions_number(&self, value: f64) -> bool {
        self.numbers.iter().any(|&seen| (seen - value).abs() < 1e-9)
    }

    pub fn mentions_text(&self, value: &str) -> bool {
        self.lowered.contains(&value.to_lowercase())
            || explicit_date(&self.original).as_deref() == Some(value)
            || clock_times(&self.original).iter().any(|time| time == value)
    }
}

pub fn numbers_in(text: &str) -> Vec<f64> {
    let mut numbers = Vec::new();
    let characters: Vec<char> = text.chars().collect();
    let mut index = 0;
    while index < characters.len() {
        if characters[index].is_ascii_digit() {
            let start = index;
            while index < characters.len()
                && (characters[index].is_ascii_digit()
                    || (matches!(characters[index], ',' | '.') && characters.get(index + 1).is_some_and(char::is_ascii_digit)))
            {
                index += 1;
            }
            let token: String = characters[start..index].iter().filter(|c| **c != ',').collect();
            if let Ok(value) = token.parse::<f64>() {
                numbers.push(value);
            }
        } else {
            index += 1;
        }
    }
    for word in text.to_lowercase().split(|c: char| !c.is_alphabetic()) {
        if let Some((_, value)) = NUMBER_WORDS.iter().find(|(name, _)| *name == word) {
            numbers.push(*value);
        }
    }
    numbers
}
