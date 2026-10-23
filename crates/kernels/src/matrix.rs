use std::sync::{Arc, OnceLock};

use cact::{Archive, ElementKind, TensorBytes, TensorRecord, TERNARY_RECORD_BITS};

use crate::codebook::Codebooks;
use crate::walsh;

const PARALLEL_ROW_THRESHOLD: usize = 256;
const PARALLEL_VECTOR_ROWS: usize = 512;
const ROWS_PER_TASK: usize = 64;
const ROW_TILE: usize = 8;

#[cfg(not(target_arch = "aarch64"))]
pub struct NaturalInput;
#[cfg(target_arch = "aarch64")]
pub use crate::panel::NaturalInput;

#[derive(Clone)]
pub struct QuantisedMatrix {
    bytes: TensorBytes,
    packed_len: usize,
    rows: usize,
    columns: usize,
    padded_columns: usize,
    group: usize,
    bits: u32,
    table: LookupTable,
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    norm_values: Arc<[f32]>,
    #[cfg(target_arch = "aarch64")]
    panel_cache: Arc<OnceLock<Option<crate::panel::PanelMatrix>>>,
}

impl QuantisedMatrix {
    pub fn from_archive(archive: &Archive, index: usize) -> Self {
        let record = archive.record(index);
        assert_eq!(record.kind, ElementKind::CodebookQuantised, "tensor {index} is not codebook-quantised");
        let codebooks = Codebooks::from_archive(archive.codebook(), record.group_size);
        Self::from_parts(archive.tensor_bytes(index), record, &codebooks)
    }

    fn from_parts(bytes: TensorBytes, record: &TensorRecord, codebooks: &Codebooks) -> Self {
        let packed_len = record.packed_bytes();
        let norm_bytes = &bytes[packed_len..];
        let norm_values: Arc<[f32]> = norm_bytes.chunks_exact(2).map(|pair| cact::half_to_f32(u16::from_le_bytes([pair[0], pair[1]]))).collect();
        Self {
            bytes,
            norm_values,
            packed_len,
            rows: record.rows(),
            columns: record.columns(),
            padded_columns: record.padded_columns(),
            group: record.group_size,
            bits: record.bits,
            table: lookup_table(record.bits, codebooks),
            #[cfg(target_arch = "aarch64")]
            panel_cache: Arc::new(OnceLock::new()),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn padded_columns(&self) -> usize {
        self.padded_columns
    }

    pub fn group(&self) -> usize {
        self.group
    }

    fn row_bytes(&self) -> usize {
        let bits = if self.bits == TERNARY_RECORD_BITS { 2 } else { self.bits as usize };
        self.padded_columns * bits / 8
    }

    fn groups_per_row(&self) -> usize {
        self.padded_columns / self.group
    }

    fn norms(&self) -> &[u16] {
        let norm_bytes = &self.bytes[self.packed_len..];
        assert_eq!(norm_bytes.as_ptr() as usize % 2, 0, "norm table is misaligned");
        unsafe { std::slice::from_raw_parts(norm_bytes.as_ptr().cast::<u16>(), norm_bytes.len() / 2) }
    }

    fn row_norms(&self, row: usize) -> &[u16] {
        let groups = self.groups_per_row();
        &self.norms()[row * groups..(row + 1) * groups]
    }

    fn row_packed(&self, row: usize) -> &[u8] {
        let stride = self.row_bytes();
        &self.bytes[row * stride..(row + 1) * stride]
    }

    #[cfg(target_arch = "aarch64")]
    pub fn panels(&self) -> Option<crate::panel::PanelMatrix> {
        self.build_panels()
    }

    #[cfg(target_arch = "aarch64")]
    fn panel_matrix(&self) -> Option<&crate::panel::PanelMatrix> {
        self.panel_cache.get_or_init(|| self.build_panels()).as_ref()
    }

    #[cfg(target_arch = "aarch64")]
    fn build_panels(&self) -> Option<crate::panel::PanelMatrix> {
        let levels = self.table.integer_levels.clone()?;
        if self.bits != 2 || !self.group.is_multiple_of(16) {
            return None;
        }
        let groups = self.groups_per_row();
        let norm_at = |row: usize, group: usize| self.norm_values[row * groups + group];
        Some(crate::panel::PanelMatrix::build(self.rows, self.padded_columns, self.group, levels, &self.bytes[..self.packed_len], self.row_bytes(), norm_at))
    }

    pub fn prepare(&self, input: &[f32], rotated: &mut Vec<f32>) {
        rotated.clear();
        rotated.extend_from_slice(input);
        rotated.resize(self.padded_columns, 0.0);
        walsh::rotate_groups(rotated, self.group);
    }

    pub fn prepare_batch(&self, inputs: &[f32], count: usize, rotated: &mut Vec<f32>) {
        let width = self.columns;
        rotated.clear();
        rotated.resize(count * self.padded_columns, 0.0);
        for (source, target) in inputs.chunks_exact(width).zip(rotated.chunks_exact_mut(self.padded_columns)) {
            target[..width].copy_from_slice(source);
            walsh::rotate_groups(target, self.group);
        }
    }

    pub fn dequantise_row(&self, row: usize, output: &mut Vec<f32>) {
        output.clear();
        output.resize(self.padded_columns, 0.0);
        let packed = self.row_packed(row);
        let norms = self.row_norms(row);
        for (group_index, group_out) in output.chunks_exact_mut(self.group).enumerate() {
            let norm = cact::half_to_f32(norms[group_index]);
            for (position, slot) in group_out.iter_mut().enumerate() {
                *slot = self.table.value(packed, group_index * self.group + position) * norm;
            }
            walsh::transform_in_place(group_out);
        }
        output.truncate(self.columns);
    }

    pub fn multiply_vector(&self, rotated: &[f32], output: &mut [f32]) {
        debug_assert_eq!(rotated.len(), self.padded_columns);
        debug_assert_eq!(output.len(), self.rows);
        let planes = self.plane_input(rotated);
        if self.rows >= PARALLEL_VECTOR_ROWS {
            let jobs: Vec<RowJob> = output.chunks_mut(ROWS_PER_TASK).enumerate().map(|(chunk, slots)| RowJob::new(chunk * ROWS_PER_TASK, slots)).collect();
            crate::pool::shared().run(jobs.len(), &|index| self.fill_rows(jobs[index].first, rotated, &planes, unsafe { jobs[index].slots() }));
        } else {
            self.fill_rows(0, rotated, &planes, output);
        }
    }

    pub fn multiply_fused(jobs: Vec<(&QuantisedMatrix, usize, &mut [f32])>, rotated: &[f32]) {
        let mut shared: Vec<((u32, usize), PlaneOption)> = Vec::new();
        for (matrix, _, _) in &jobs {
            let key = (matrix.bits, matrix.group);
            if !shared.iter().any(|(known, _)| *known == key) {
                shared.push((key, matrix.plane_input(rotated)));
            }
        }
        let mut tasks = Vec::new();
        for (matrix, first_row, output) in jobs {
            let key = (matrix.bits, matrix.group);
            let plane = &shared.iter().find(|(known, _)| *known == key).expect("plane prepared for every format").1;
            for (chunk, slots) in output.chunks_mut(ROWS_PER_TASK).enumerate() {
                tasks.push((matrix, plane, first_row + chunk * ROWS_PER_TASK, slots));
            }
        }
        if tasks.iter().map(|task| task.3.len()).sum::<usize>() >= PARALLEL_VECTOR_ROWS {
            let jobs: Vec<(&QuantisedMatrix, &PlaneOption, RowJob)> =
                tasks.into_iter().map(|(matrix, plane, first, slots)| (matrix, plane, RowJob::new(first, slots))).collect();
            crate::pool::shared().run(jobs.len(), &|index| {
                let (matrix, plane, job) = &jobs[index];
                matrix.fill_rows(job.first, rotated, plane, unsafe { job.slots() });
            });
        } else {
            tasks.into_iter().for_each(|(matrix, plane, first, slots)| matrix.fill_rows(first, rotated, plane, slots));
        }
    }

    pub fn multiply_rows(&self, first_row: usize, rotated: &[f32], output: &mut [f32]) {
        let planes = self.plane_input(rotated);
        if output.len() >= PARALLEL_VECTOR_ROWS {
            crate::pool::parallel_chunks_mut(output, ROWS_PER_TASK, &|chunk, slots| self.fill_rows(first_row + chunk * ROWS_PER_TASK, rotated, &planes, slots));
        } else {
            self.fill_rows(first_row, rotated, &planes, output);
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn plane_input(&self, rotated: &[f32]) -> PlaneOption {
        let enabled = integer_kernels_enabled();
        if !enabled || self.table.integer_levels.is_none() {
            return None;
        }
        if self.panel_matrix().is_some() {
            return Some(Prepared::Natural(crate::panel::NaturalInput::new(rotated, 1, self.padded_columns, self.group)));
        }
        crate::neon::supports(self.table.bits, self.group).then(|| Prepared::Plane(crate::neon::PlaneInput::new(rotated, self.table.bits, self.group)))
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn plane_input(&self, _rotated: &[f32]) -> Option<()> {
        None
    }

    #[cfg(target_arch = "aarch64")]
    fn multiply_batch_panels(&self, panels: &crate::panel::PanelMatrix, rotated: &[f32], count: usize, output: &mut [f32]) {
        const PANELS_PER_TASK: usize = 16;
        let input = crate::panel::NaturalInput::new(rotated, count, self.padded_columns, self.group);
        let total = panels.panels();
        let rows_per_panel = crate::panel::PanelMatrix::panel_rows();
        let parts: Vec<(usize, usize, Vec<f32>)> = crate::pool::parallel_map(total.div_ceil(PANELS_PER_TASK), &|task| {
            let first = task * PANELS_PER_TASK;
            let take = PANELS_PER_TASK.min(total - first);
            let mut buffer = vec![0.0f32; count * take * rows_per_panel];
            panels.multiply_panels(&input, count, first, take, &mut buffer);
            (first, take * rows_per_panel, buffer)
        });
        for (first, stride, buffer) in parts {
            let row_start = first * rows_per_panel;
            let usable = stride.min(self.rows - row_start);
            for token in 0..count {
                output[token * self.rows + row_start..token * self.rows + row_start + usable].copy_from_slice(&buffer[token * stride..token * stride + usable]);
            }
        }
    }

    pub fn multiply_batch_fused(jobs: Vec<(&QuantisedMatrix, &mut [f32])>, rotated: &[f32], count: usize) {
        #[cfg(target_arch = "aarch64")]
        if integer_kernels_enabled() {
            let panels: Option<Vec<&crate::panel::PanelMatrix>> = jobs.iter().map(|(matrix, _)| matrix.panel_matrix()).collect();
            let shared = jobs.first().map(|(matrix, _)| (matrix.padded_columns, matrix.group));
            if let (Some(panels), Some((columns, group))) = (panels, shared) {
                if jobs.iter().all(|(matrix, _)| (matrix.padded_columns, matrix.group) == (columns, group)) {
                    return Self::multiply_panels_fused(jobs, &panels, rotated, count, columns, group);
                }
            }
        }
        for (matrix, output) in jobs {
            matrix.multiply_batch(rotated, count, output);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn natural_batch(&self, _inputs: &[f32], _count: usize, _prepare_row: &(dyn Fn(usize, &mut [f32]) + Sync)) -> Option<NaturalInput> {
        None
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn multiply_natural_fused(_jobs: Vec<(&QuantisedMatrix, &mut [f32])>, _input: &NaturalInput, _count: usize) {
        unreachable!("natural inputs exist only where panels do")
    }

    #[cfg(target_arch = "aarch64")]
    pub fn natural_batch(&self, inputs: &[f32], count: usize, prepare_row: &(dyn Fn(usize, &mut [f32]) + Sync)) -> Option<crate::panel::NaturalInput> {
        let enabled = integer_kernels_enabled();
        (enabled && self.panel_matrix().is_some())
            .then(|| crate::panel::NaturalInput::build(inputs, count, self.columns, self.padded_columns, self.group, prepare_row))
    }

    #[cfg(target_arch = "aarch64")]
    pub fn multiply_natural_fused(jobs: Vec<(&QuantisedMatrix, &mut [f32])>, input: &crate::panel::NaturalInput, count: usize) {
        let panels: Vec<&crate::panel::PanelMatrix> =
            jobs.iter().map(|(matrix, _)| matrix.panel_matrix().expect("natural input needs panel matrices")).collect();
        Self::multiply_panels_prepared(jobs, &panels, input, count);
    }

    #[cfg(target_arch = "aarch64")]
    fn multiply_panels_fused(
        jobs: Vec<(&QuantisedMatrix, &mut [f32])>,
        panels: &[&crate::panel::PanelMatrix],
        rotated: &[f32],
        count: usize,
        columns: usize,
        group: usize,
    ) {
        let input = crate::panel::NaturalInput::new(rotated, count, columns, group);
        Self::multiply_panels_prepared(jobs, panels, &input, count);
    }

    #[cfg(target_arch = "aarch64")]
    fn multiply_panels_prepared(
        jobs: Vec<(&QuantisedMatrix, &mut [f32])>,
        panels: &[&crate::panel::PanelMatrix],
        input: &crate::panel::NaturalInput,
        count: usize,
    ) {
        const PANELS_PER_TASK: usize = 16;
        let rows_per_panel = crate::panel::PanelMatrix::panel_rows();
        let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
        for (job, matrix_panels) in panels.iter().enumerate() {
            let total = matrix_panels.panels();
            for first in (0..total).step_by(PANELS_PER_TASK) {
                tasks.push((job, first, PANELS_PER_TASK.min(total - first)));
            }
        }
        let parts: Vec<Vec<f32>> = crate::pool::parallel_map(tasks.len(), &|index| {
            let (job, first, take) = tasks[index];
            let mut buffer = vec![0.0f32; count * take * rows_per_panel];
            panels[job].multiply_panels(input, count, first, take, &mut buffer);
            buffer
        });
        let mut outputs: Vec<(usize, &mut [f32])> = jobs.into_iter().map(|(matrix, output)| (matrix.rows, output)).collect();
        for ((job, first, take), buffer) in tasks.into_iter().zip(parts) {
            let (rows, output) = &mut outputs[job];
            let stride = take * rows_per_panel;
            let row_start = first * rows_per_panel;
            let usable = stride.min(*rows - row_start);
            for token in 0..count {
                output[token * *rows + row_start..token * *rows + row_start + usable].copy_from_slice(&buffer[token * stride..token * stride + usable]);
            }
        }
    }

    pub fn multiply_batch(&self, rotated: &[f32], count: usize, output: &mut [f32]) {
        debug_assert_eq!(rotated.len(), count * self.padded_columns);
        debug_assert_eq!(output.len(), count * self.rows);
        #[cfg(target_arch = "aarch64")]
        if integer_kernels_enabled() {
            if let Some(panels) = self.panel_matrix() {
                return self.multiply_batch_panels(panels, rotated, count, output);
            }
        }
        let width = self.padded_columns;
        let rows = self.rows;
        #[cfg(target_arch = "aarch64")]
        let planes: Vec<Option<crate::neon::PlaneInput>> = if count > 0 && self.plane_input(&rotated[..width]).is_some() {
            crate::pool::parallel_map(count, &|token| self.plane_input(&rotated[token * width..(token + 1) * width]).and_then(Prepared::into_plane))
        } else {
            Vec::new()
        };
        let compute = |first: usize, slots: &mut [f32]| {
            let tile_rows = slots.len() / count;
            #[cfg(target_arch = "aarch64")]
            if let (false, Some(levels)) = (planes.is_empty(), &self.table.integer_levels) {
                let groups = self.groups_per_row();
                for token in 0..count {
                    let input = planes[token].as_ref().expect("planes exist for every token");
                    for offset in 0..tile_rows {
                        let row = first + offset;
                        slots[offset * count + token] =
                            crate::neon::row_dot(self.row_packed(row), &self.norm_values[row * groups..(row + 1) * groups], input, levels);
                    }
                }
                return;
            }
            let mut tile = vec![0.0f32; ROW_TILE * width];
            let mut done = 0;
            while done < tile_rows {
                let take = ROW_TILE.min(tile_rows - done);
                for offset in 0..take {
                    self.expand_row_scaled(first + done + offset, &mut tile[offset * width..(offset + 1) * width]);
                }
                for token in 0..count {
                    let input = &rotated[token * width..(token + 1) * width];
                    for offset in 0..take {
                        slots[(done + offset) * count + token] = dot(&tile[offset * width..(offset + 1) * width], input);
                    }
                }
                done += take;
            }
        };
        let mut transposed = vec![0.0f32; rows * count];
        if rows >= PARALLEL_ROW_THRESHOLD {
            crate::pool::parallel_chunks_mut(&mut transposed, ROWS_PER_TASK * count, &|chunk, slots| compute(chunk * ROWS_PER_TASK, slots));
        } else {
            compute(0, &mut transposed);
        }
        for token in 0..count {
            for row in 0..rows {
                output[token * rows + row] = transposed[row * count + token];
            }
        }
    }

    fn expand_row_scaled(&self, row: usize, scratch: &mut [f32]) {
        let packed = self.row_packed(row);
        let norms = self.row_norms(row);
        for (group_index, group_out) in scratch.chunks_exact_mut(self.group).enumerate() {
            self.table.expand_group(packed, group_index, self.group, group_out);
            let norm = cact::half_to_f32(norms[group_index]);
            group_out.iter_mut().for_each(|value| *value *= norm);
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn fill_rows(&self, first_row: usize, rotated: &[f32], planes: &PlaneOption, slots: &mut [f32]) {
        if let (Some(Prepared::Natural(input)), Some(panels)) = (planes, self.panel_matrix()) {
            if first_row.is_multiple_of(crate::panel::PanelMatrix::panel_rows()) {
                let first_panel = first_row / crate::panel::PanelMatrix::panel_rows();
                panels.multiply_panels(input, 1, first_panel, slots.len().div_ceil(crate::panel::PanelMatrix::panel_rows()), slots);
                return;
            }
            return self.fill_rows_float(first_row, rotated, slots);
        }
        if let (Some(Prepared::Plane(planes)), Some(levels)) = (planes, &self.table.integer_levels) {
            let groups = self.groups_per_row();
            for (offset, slot) in slots.iter_mut().enumerate() {
                let row = first_row + offset;
                *slot = crate::neon::row_dot(self.row_packed(row), &self.norm_values[row * groups..(row + 1) * groups], planes, levels);
            }
            return;
        }
        self.fill_rows_float(first_row, rotated, slots);
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn fill_rows(&self, first_row: usize, rotated: &[f32], _planes: &Option<()>, slots: &mut [f32]) {
        self.fill_rows_float(first_row, rotated, slots);
    }

    fn fill_rows_float(&self, first_row: usize, rotated: &[f32], slots: &mut [f32]) {
        for (offset, slot) in slots.iter_mut().enumerate() {
            let row = first_row + offset;
            let packed = self.row_packed(row);
            let norms = self.row_norms(row);
            let mut total = 0.0f32;
            for (group_index, norm) in norms.iter().enumerate() {
                let start = group_index * self.group;
                let partial = self.table.dot_group(packed, group_index, self.group, &rotated[start..start + self.group]);
                total += partial * cact::half_to_f32(*norm);
            }
            *slot = total;
        }
    }
}

#[cfg(target_arch = "aarch64")]
enum Prepared {
    Natural(crate::panel::NaturalInput),
    Plane(crate::neon::PlaneInput),
}

#[cfg(target_arch = "aarch64")]
impl Prepared {
    fn into_plane(self) -> Option<crate::neon::PlaneInput> {
        match self {
            Prepared::Plane(plane) => Some(plane),
            Prepared::Natural(_) => None,
        }
    }
}

#[cfg(target_arch = "aarch64")]
type PlaneOption = Option<Prepared>;
#[cfg(not(target_arch = "aarch64"))]
type PlaneOption = Option<()>;

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn integer_kernels_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("RUSTYNEEDLE_FLOAT_KERNELS").is_none())
}

#[derive(Clone, Copy)]
struct RowJob {
    first: usize,
    start: *mut f32,
    length: usize,
}

unsafe impl Sync for RowJob {}

impl RowJob {
    fn new(first: usize, slots: &mut [f32]) -> Self {
        Self { first, start: slots.as_mut_ptr(), length: slots.len() }
    }

    unsafe fn slots<'a>(self) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(self.start, self.length)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    for (x, y) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        for lane in 0..8 {
            lanes[lane] += x[lane] * y[lane];
        }
    }
    lanes.iter().sum()
}

fn lookup_table(bits: u32, codebooks: &Codebooks) -> LookupTable {
    match bits {
        1 => LookupTable::new(1, pad16(&codebooks.binary())),
        2 => LookupTable::new(2, pad16(codebooks.two_bit())),
        3 => LookupTable::new(3, pad16(codebooks.three_bit())),
        4 => LookupTable::new(4, pad16(codebooks.four_bit())),
        TERNARY_RECORD_BITS => LookupTable::new(2, pad16(&codebooks.ternary_by_crumb())),
        other => panic!("unsupported codebook width {other}"),
    }
}

fn pad16(values: &[f32]) -> [f32; 16] {
    let mut out = [0.0; 16];
    out[..values.len()].copy_from_slice(values);
    out
}

#[derive(Clone)]
enum ByteTables {
    Eight(Vec<[f32; 8]>),
    Four(Vec<[f32; 4]>),
    Two(Vec<[f32; 2]>),
    Bitwise,
}

fn build_tables<const N: usize>(bits: u32, levels: &[f32; 16]) -> Vec<[f32; N]> {
    let mask = (1usize << bits) - 1;
    (0..256usize)
        .map(|byte| {
            let mut entry = [0.0f32; N];
            for (slot, value) in entry.iter_mut().enumerate() {
                *value = levels[(byte >> (slot * bits as usize)) & mask];
            }
            entry
        })
        .collect()
}

fn expand_with<const N: usize>(tables: &[[f32; N]], bytes: &[u8], out: &mut [f32]) {
    for (chunk, byte) in out.chunks_exact_mut(N).zip(bytes) {
        chunk.copy_from_slice(&tables[*byte as usize]);
    }
}

fn dot_with<const N: usize>(tables: &[[f32; N]], bytes: &[u8], inputs: &[f32]) -> f32 {
    let mut lanes = [0.0f32; N];
    for (byte, chunk) in bytes.iter().zip(inputs.chunks_exact(N)) {
        let entry = &tables[*byte as usize];
        for lane in 0..N {
            lanes[lane] += entry[lane] * chunk[lane];
        }
    }
    lanes.iter().sum()
}

#[derive(Clone)]
struct LookupTable {
    bits: u32,
    levels: [f32; 16],
    tables: ByteTables,
    #[cfg(target_arch = "aarch64")]
    integer_levels: Option<crate::neon::IntegerLevels>,
}

impl LookupTable {
    fn new(bits: u32, levels: [f32; 16]) -> Self {
        let tables = match bits {
            1 => ByteTables::Eight(build_tables(bits, &levels)),
            2 => ByteTables::Four(build_tables(bits, &levels)),
            4 => ByteTables::Two(build_tables(bits, &levels)),
            _ => ByteTables::Bitwise,
        };
        #[cfg(target_arch = "aarch64")]
        let integer_levels = matches!(bits, 2 | 4).then(|| crate::neon::IntegerLevels::new(&levels, 1usize << bits));
        Self {
            bits,
            levels,
            tables,
            #[cfg(target_arch = "aarch64")]
            integer_levels,
        }
    }

    fn per_byte(&self) -> usize {
        match self.tables {
            ByteTables::Eight(_) => 8,
            ByteTables::Four(_) => 4,
            ByteTables::Two(_) => 2,
            ByteTables::Bitwise => 0,
        }
    }

    fn group_bytes<'a>(&self, packed: &'a [u8], group_index: usize, group: usize) -> &'a [u8] {
        let bytes = group / self.per_byte();
        &packed[group_index * bytes..(group_index + 1) * bytes]
    }

    fn expand_group(&self, packed: &[u8], group_index: usize, group: usize, out: &mut [f32]) {
        match &self.tables {
            ByteTables::Eight(tables) => expand_with(tables, self.group_bytes(packed, group_index, group), out),
            ByteTables::Four(tables) => expand_with(tables, self.group_bytes(packed, group_index, group), out),
            ByteTables::Two(tables) => expand_with(tables, self.group_bytes(packed, group_index, group), out),
            ByteTables::Bitwise => {
                let start = group_index * group;
                for (position, slot) in out.iter_mut().enumerate() {
                    *slot = self.value(packed, start + position);
                }
            }
        }
    }

    fn dot_group(&self, packed: &[u8], group_index: usize, group: usize, inputs: &[f32]) -> f32 {
        match &self.tables {
            ByteTables::Eight(tables) => dot_with(tables, self.group_bytes(packed, group_index, group), inputs),
            ByteTables::Four(tables) => dot_with(tables, self.group_bytes(packed, group_index, group), inputs),
            ByteTables::Two(tables) => dot_with(tables, self.group_bytes(packed, group_index, group), inputs),
            ByteTables::Bitwise => {
                let start = group_index * group;
                (0..group).map(|position| self.value(packed, start + position) * inputs[position]).sum()
            }
        }
    }

    #[inline(always)]
    fn value(&self, packed: &[u8], index: usize) -> f32 {
        let bit_offset = index * self.bits as usize;
        let byte = bit_offset / 8;
        let shift = bit_offset % 8;
        let mask = (1u32 << self.bits) - 1;
        let raw =
            if shift + self.bits as usize > 8 { ((packed[byte] as u32) | (packed[byte + 1] as u32) << 8) >> shift } else { (packed[byte] as u32) >> shift };
        self.levels[(raw & mask) as usize]
    }
}
