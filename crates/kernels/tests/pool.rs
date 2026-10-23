use std::sync::atomic::{AtomicUsize, Ordering};

use rustyneedle_kernels::spin_pool;

#[test]
fn every_task_runs_exactly_once_across_many_differently_sized_runs() {
    for round in 0..200_000usize {
        let tasks = [513, 2, 128, 3, 64, 2, 7, 129, 2, 3][round % 10];
        let counters: Vec<AtomicUsize> = (0..tasks).map(|_| AtomicUsize::new(0)).collect();
        spin_pool().run(tasks, &|index| {
            counters[index].fetch_add(1, Ordering::Relaxed);
        });
        assert!(counters.iter().all(|counter| counter.load(Ordering::Relaxed) == 1), "round {round} with {tasks} tasks");
    }
}

#[test]
fn chunked_writes_cover_the_whole_slice() {
    let mut data = vec![0u32; 1000];
    rustyneedle_kernels::parallel_chunks_mut(&mut data, 7, &|chunk, slice| slice.iter_mut().for_each(|value| *value = chunk as u32 + 1));
    assert!(data.iter().enumerate().all(|(index, value)| *value == (index / 7) as u32 + 1));
}

#[test]
fn a_panicking_task_is_reported_to_the_caller_and_the_pool_keeps_working() {
    let outcome = std::panic::catch_unwind(|| {
        spin_pool().run(64, &|index| {
            if index == 17 {
                panic!("deliberate");
            }
        });
    });
    assert!(outcome.is_err());
    let counter = AtomicUsize::new(0);
    spin_pool().run(64, &|_| {
        counter.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(counter.load(Ordering::Relaxed), 64);
}
