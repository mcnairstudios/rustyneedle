use std::fmt;
use std::time::Instant;

pub struct Timing {
    pub median_ms: f64,
    pub best_ms: f64,
}

impl fmt::Display for Timing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "median {:.2} ms, best {:.2} ms", self.median_ms, self.best_ms)
    }
}

pub fn measure(rounds: usize, mut work: impl FnMut()) -> Timing {
    work();
    let mut samples: Vec<f64> = (0..rounds)
        .map(|_| {
            let started = Instant::now();
            work();
            started.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    samples.sort_by(|a, b| a.total_cmp(b));
    Timing { median_ms: samples[samples.len() / 2], best_ms: samples[0] }
}

pub fn token_ids(count: usize, stride: u32) -> Vec<u32> {
    (0..count as u32).map(|index| 10 + (index * stride) % 8000).collect()
}
