use tokenizer::Tokenizer;

use crate::Decoded;

const FRAME_SECONDS: f64 = 0.08;
const FILTER_WIDTH: usize = 7;
const SPACE_MARK: char = '\u{2581}';

#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub probability: f32,
}

fn standardise(row: &mut [f32]) {
    let mean = row.iter().sum::<f32>() / row.len() as f32;
    let variance = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / row.len() as f32;
    let scale = 1.0 / (variance.sqrt() + 1e-6);
    row.iter_mut().for_each(|v| *v = (*v - mean) * scale);
}

fn median_filter(row: &[f32]) -> Vec<f32> {
    let half = FILTER_WIDTH / 2;
    let length = row.len() as isize;
    let reflect = |index: isize| -> f32 {
        let mut i = index;
        if i < 0 {
            i = -i;
        }
        if i >= length {
            i = 2 * (length - 1) - i;
        }
        row[i.clamp(0, length - 1) as usize]
    };
    (0..length)
        .map(|centre| {
            let mut window: Vec<f32> = (-(half as isize)..=half as isize).map(|offset| reflect(centre + offset)).collect();
            window.sort_by(f32::total_cmp);
            window[half]
        })
        .collect()
}

fn dtw_jump_frames(matrix: &[Vec<f32>]) -> Vec<usize> {
    let (rows, columns) = (matrix.len(), matrix[0].len());
    let mut cost = vec![vec![f32::INFINITY; columns + 1]; rows + 1];
    let mut trace = vec![vec![0u8; columns + 1]; rows + 1];
    cost[0][0] = 0.0;
    for i in 1..=rows {
        for j in 1..=columns {
            let diagonal = cost[i - 1][j - 1];
            let up = cost[i - 1][j];
            let left = cost[i][j - 1];
            let (best, direction) = if diagonal <= up && diagonal <= left {
                (diagonal, 0)
            } else if up <= left {
                (up, 1)
            } else {
                (left, 2)
            };
            cost[i][j] = -matrix[i - 1][j - 1] + best;
            trace[i][j] = direction;
        }
    }
    let (mut i, mut j) = (rows, columns);
    let mut path = Vec::new();
    while i > 0 || j > 0 {
        path.push((i.saturating_sub(1), j.saturating_sub(1)));
        match trace[i][j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => i -= 1,
            _ => j -= 1,
        }
        if i == 0 && j > 0 {
            j = 0;
        }
    }
    path.reverse();
    let mut jumps = vec![0usize; rows];
    let mut seen = vec![false; rows];
    for (row, column) in path {
        if !seen[row] {
            seen[row] = true;
            jumps[row] = column;
        }
    }
    jumps
}

pub(crate) fn words(decoded: &Decoded, tokenizer: &Tokenizer) -> Vec<Word> {
    let tokens = decoded.ids.len();
    if tokens == 0 || decoded.frames == 0 {
        return Vec::new();
    }
    let vocabulary = tokenizer.vocabulary();
    let mut matrix: Vec<Vec<f32>> = (0..=tokens).map(|row| decoded.alignment[row.min(decoded.alignment.len() - 1)].clone()).collect();
    for row in matrix.iter_mut() {
        standardise(row);
        *row = median_filter(row);
    }
    let jumps = dtw_jump_frames(&matrix);
    let mut starts = Vec::new();
    for (index, &id) in decoded.ids.iter().enumerate() {
        if index == 0 || vocabulary.piece(id).starts_with(SPACE_MARK) {
            starts.push(index);
        }
    }
    starts.push(tokens);
    let mut boundaries: Vec<usize> = starts.iter().map(|&token| jumps[token]).collect();
    for index in 1..boundaries.len() {
        if boundaries[index] <= boundaries[index - 1] {
            boundaries[index] = (boundaries[index - 1] + 1).min(decoded.frames);
        }
    }
    starts
        .windows(2)
        .enumerate()
        .map(|(word, bounds)| {
            let (first, last) = (bounds[0], bounds[1]);
            let text = tokenizer.decode(&decoded.ids[first..last]).trim().to_owned();
            let probability = decoded.probabilities[first..last].iter().product::<f32>().powf(1.0 / (last - first) as f32);
            Word { word: text, start: boundaries[word] as f64 * FRAME_SECONDS, end: boundaries[word + 1] as f64 * FRAME_SECONDS, probability }
        })
        .collect()
}
