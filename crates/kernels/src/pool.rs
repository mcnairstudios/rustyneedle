use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

const MAX_WORKERS: usize = 5;
const SPIN_BEFORE_PARK: Duration = Duration::from_micros(300);
const PARK_INTERVAL: Duration = Duration::from_millis(50);

type Task<'a> = &'a (dyn Fn(usize) + Sync);

const FIELD_BITS: u32 = 22;
const FIELD_MASK: u64 = (1 << FIELD_BITS) - 1;
const MAX_TASKS: usize = 1 << FIELD_BITS;
const RUN_SHIFT: u32 = 2 * FIELD_BITS;

const SPINS_PER_CLOCK_CHECK: u32 = 256;

#[cfg(target_os = "macos")]
fn prefer_performance_cores() {
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
fn prefer_performance_cores() {}

struct Shared {
    panicked: AtomicBool,
    sleepers: AtomicUsize,
    cursor: AtomicU64,
    pending: AtomicUsize,
    task: AtomicPtr<()>,
}

pub struct SpinPool {
    shared: Arc<Shared>,
    workers: Vec<Thread>,
    gate: Mutex<()>,
}

fn worker_count() -> usize {
    let requested = std::env::var("RUSTYNEEDLE_THREADS").ok().and_then(|value| value.parse::<usize>().ok());
    let available = thread::available_parallelism().map_or(1, |count| count.get());
    requested.unwrap_or(available.saturating_sub(1)).saturating_sub(1).min(MAX_WORKERS.max(requested.unwrap_or(0)))
}

fn index_of(word: u64) -> u64 {
    word & FIELD_MASK
}

fn total_of(word: u64) -> u64 {
    (word >> FIELD_BITS) & FIELD_MASK
}

fn has_work(shared: &Shared) -> bool {
    let word = shared.cursor.load(Ordering::Acquire);
    index_of(word) < total_of(word)
}

fn claim_and_run(shared: &Shared) {
    loop {
        let word = shared.cursor.load(Ordering::Acquire);
        if index_of(word) >= total_of(word) {
            return;
        }
        if shared.cursor.compare_exchange_weak(word, word + 1, Ordering::AcqRel, Ordering::Relaxed).is_err() {
            continue;
        }
        let task = unsafe { *(shared.task.load(Ordering::Acquire) as *const Task<'static>) };
        if catch_unwind(AssertUnwindSafe(|| task(index_of(word) as usize))).is_err() {
            shared.panicked.store(true, Ordering::Release);
        }
        shared.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

fn work(shared: Arc<Shared>) {
    prefer_performance_cores();
    let mut idle_since = Instant::now();
    let mut spins = 0u32;
    loop {
        if has_work(&shared) {
            claim_and_run(&shared);
            idle_since = Instant::now();
            spins = 0;
            continue;
        }
        spins += 1;
        if spins < SPINS_PER_CLOCK_CHECK {
            std::hint::spin_loop();
            continue;
        }
        spins = 0;
        if idle_since.elapsed() < SPIN_BEFORE_PARK {
            continue;
        }
        shared.sleepers.fetch_add(1, Ordering::SeqCst);
        if !has_work_sequential(&shared) {
            thread::park_timeout(PARK_INTERVAL);
        }
        shared.sleepers.fetch_sub(1, Ordering::SeqCst);
        idle_since = Instant::now();
    }
}

fn has_work_sequential(shared: &Shared) -> bool {
    let word = shared.cursor.load(Ordering::SeqCst);
    index_of(word) < total_of(word)
}

impl SpinPool {
    fn start() -> Self {
        let shared = Arc::new(Shared {
            panicked: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            cursor: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            task: AtomicPtr::new(std::ptr::null_mut()),
        });
        let workers = (0..worker_count())
            .map(|_| {
                let shared = Arc::clone(&shared);
                let handle = thread::Builder::new().name("needle-pool".into()).spawn(move || work(shared)).expect("spawn pool worker");
                handle.thread().clone()
            })
            .collect();
        Self { shared, workers, gate: Mutex::new(()) }
    }

    pub fn run(&self, tasks: usize, task: Task<'_>) {
        let Ok(_guard) = self.gate.try_lock() else {
            return (0..tasks).for_each(task);
        };
        if self.workers.is_empty() || tasks < 2 {
            return (0..tasks).for_each(task);
        }
        assert!(tasks < MAX_TASKS, "too many tasks for one run");
        let shared = &self.shared;
        let published: Task<'_> = task;
        shared.task.store(&published as *const Task<'_> as *mut (), Ordering::Release);
        shared.pending.store(tasks, Ordering::Release);
        let run = (shared.cursor.load(Ordering::Acquire) >> RUN_SHIFT) + 1;
        shared.cursor.store((run << RUN_SHIFT) | ((tasks as u64) << FIELD_BITS), Ordering::SeqCst);
        if shared.sleepers.load(Ordering::SeqCst) > 0 {
            self.workers.iter().for_each(Thread::unpark);
        }
        claim_and_run(shared);
        while shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        if shared.panicked.swap(false, Ordering::AcqRel) {
            resume_unwind(Box::new("a pool task panicked"));
        }
    }
}

struct Partition<T> {
    start: *mut T,
    length: usize,
    chunk: usize,
}

unsafe impl<T: Send> Sync for Partition<T> {}

impl SpinPool {
    pub fn for_each_chunk<T: Send>(&self, data: &mut [T], chunk: usize, task: &(dyn Fn(usize, &mut [T]) + Sync)) {
        let partition = Partition { start: data.as_mut_ptr(), length: data.len(), chunk };
        let chunks = data.len().div_ceil(chunk);
        let partition = &partition;
        self.run(chunks, &|index| {
            let begin = index * partition.chunk;
            let end = (begin + partition.chunk).min(partition.length);
            task(index, unsafe { std::slice::from_raw_parts_mut(partition.start.add(begin), end - begin) });
        });
    }
}

type PairTask<'a, A, B> = &'a (dyn Fn(usize, &mut [A], &mut [B]) + Sync);

pub fn parallel_chunks_pair_mut<A: Send, B: Send>(first: &mut [A], first_chunk: usize, second: &mut [B], second_chunk: usize, task: PairTask<'_, A, B>) {
    let chunks = first.len().div_ceil(first_chunk);
    assert_eq!(chunks, second.len().div_ceil(second_chunk), "both outputs must split into the same number of chunks");
    let (a, b) = (
        Partition { start: first.as_mut_ptr(), length: first.len(), chunk: first_chunk },
        Partition { start: second.as_mut_ptr(), length: second.len(), chunk: second_chunk },
    );
    let (a, b) = (&a, &b);
    shared().run(chunks, &|index| {
        let (begin_a, begin_b) = (index * a.chunk, index * b.chunk);
        let (end_a, end_b) = ((begin_a + a.chunk).min(a.length), (begin_b + b.chunk).min(b.length));
        task(index, unsafe { std::slice::from_raw_parts_mut(a.start.add(begin_a), end_a - begin_a) }, unsafe {
            std::slice::from_raw_parts_mut(b.start.add(begin_b), end_b - begin_b)
        });
    });
}

pub const PARALLEL_MIN_ITEMS: usize = 32;

pub fn parallel_map<T: Send>(count: usize, task: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
    let mut results: Vec<Option<T>> = (0..count).map(|_| None).collect();
    shared().for_each_chunk(&mut results, 1, &|index, slot| slot[0] = Some(task(index)));
    results.into_iter().map(|value| value.expect("every task produced a result")).collect()
}

pub fn parallel_chunks_mut<T: Send>(data: &mut [T], chunk: usize, task: &(dyn Fn(usize, &mut [T]) + Sync)) {
    shared().for_each_chunk(data, chunk, task);
}

pub fn shared() -> &'static SpinPool {
    static POOL: OnceLock<SpinPool> = OnceLock::new();
    POOL.get_or_init(SpinPool::start)
}
