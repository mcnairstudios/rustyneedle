pub fn order(layers: usize) -> Vec<usize> {
    if layers <= 1 {
        return (0..layers).collect();
    }
    let mut selected = vec![0, layers - 1];
    let mut order = selected.clone();
    while order.len() < layers {
        selected.sort_unstable();
        let (_, left, right) = selected
            .windows(2)
            .filter(|pair| pair[1] - pair[0] > 1)
            .map(|pair| (pair[1] - pair[0], pair[0], pair[1]))
            .max_by_key(|&(gap, left, _)| (gap, std::cmp::Reverse(left)))
            .expect("a gap remains while layers are unselected");
        let candidate = (left + right) / 2;
        selected.push(candidate);
        order.push(candidate);
    }
    order
}

pub fn select(layers: usize, depth: usize) -> Option<Vec<usize>> {
    if depth < 2.min(layers) || depth > layers {
        return None;
    }
    let mut chosen: Vec<usize> = order(layers).into_iter().take(depth).collect();
    chosen.sort_unstable();
    Some(chosen)
}
