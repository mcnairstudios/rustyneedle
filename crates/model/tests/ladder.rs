use rustyneedle_model::ladder;

#[test]
fn twenty_layers_follow_the_published_bisection_order() {
    assert_eq!(ladder::order(20), vec![0, 19, 9, 14, 4, 6, 11, 16, 2, 7, 12, 17, 1, 3, 5, 8, 10, 13, 15, 18]);
}

#[test]
fn each_depth_keeps_the_first_n_blocks_in_ascending_order() {
    assert_eq!(ladder::select(20, 2).unwrap(), vec![0, 19]);
    assert_eq!(ladder::select(20, 4).unwrap(), vec![0, 9, 14, 19]);
    assert_eq!(ladder::select(20, 8).unwrap(), vec![0, 4, 6, 9, 11, 14, 16, 19]);
    assert_eq!(ladder::select(20, 12).unwrap(), vec![0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19]);
    assert_eq!(ladder::select(20, 20).unwrap(), (0..20).collect::<Vec<_>>());
}

#[test]
fn out_of_range_depths_are_rejected() {
    assert!(ladder::select(20, 1).is_none());
    assert!(ladder::select(20, 21).is_none());
}

#[test]
fn whistle_sized_decoders_nest() {
    let eight = ladder::select(8, 8).unwrap();
    let four = ladder::select(8, 4).unwrap();
    assert!(four.iter().all(|layer| eight.contains(layer)));
    assert_eq!(ladder::select(8, 2).unwrap(), vec![0, 7]);
}
