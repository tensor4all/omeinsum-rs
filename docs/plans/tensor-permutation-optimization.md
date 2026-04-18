# Tensor Permutation Optimization Plan

## Problem Statement
`permute_data` is the bottleneck in tensor contraction. Current benchmark:
- Rust: 36.7s
- Julia: 16.4s (2.2x faster)

## Literature Review

### Key Papers
1. **Lyakh (2015)** - "Efficient tensor transpose for multicore CPU" ([OSTI](https://www.osti.gov/pages/servlets/purl/1185465))
2. **cuTT** (Hynninen & Lyakh, 2017) - GPU tensor transpose ([arXiv:1705.01598](https://arxiv.org/abs/1705.01598))
3. **TTC** - Tensor Transpose Compiler with auto-tuning
4. **EITHOT** - Permutation decomposition method

### Key Insights from Literature

1. **Leading Dimension Optimization**
   > "If leading dimensions of both tensors coincide and aggregate size exceeds cache line length, cache-efficient copy is possible"

2. **Non-uniform Partitioning**
   > "Uniform tiling fails for high-D tensors. Must group/segment specific dimensions depending on permutation pattern"

3. **Permutation Decomposition**
   > "Factorize N-dim permutation into sequence of lower-order transpositions"

4. **Cache-Oblivious Recursive**
   > "Divide-and-conquer until subtensors fit in cache, automatically adapts to cache hierarchy"

---

## Implementation Plan

### Phase 1: Analysis & Profiling
- [ ] Add instrumentation to track permutation patterns in real workloads
- [ ] Identify most common permutation types (transpose, cyclic, etc.)
- [ ] Measure cache miss rates for current implementation

### Phase 2: Dimension Analysis (Lyakh's Method)
Key idea: Classify dimensions into groups based on permutation pattern.

```
For permutation perm on tensor with shape:
1. Find "Minor Merged" (MM): leading dims that stay contiguous in output
2. Find "Major Merged" (MJ): trailing dims that stay contiguous in output
3. Find "Split" dims: dimensions that get interleaved
```

**Algorithm:**
```rust
struct PermutationAnalysis {
    // Dimensions that form contiguous chunks in BOTH input and output
    minor_merged_in: Range<usize>,  // Leading dims in input
    minor_merged_out: Range<usize>, // Where they map in output

    // Dimensions that form trailing contiguous chunks
    major_merged_in: Range<usize>,
    major_merged_out: Range<usize>,

    // Remaining dimensions that need element-wise permutation
    split_dims: Vec<usize>,
}

fn analyze_permutation(perm: &[usize]) -> PermutationAnalysis {
    // Find longest prefix where perm[i] = perm[0] + i (consecutive in output)
    // Find longest suffix where perm[n-1-i] = perm[n-1] - i
    // Middle dimensions are "split"
}
```

### Phase 3: Tiled Copy with Dimension Grouping

**For each permutation type, use optimal strategy:**

| Pattern | Strategy | Inner Loop |
|---------|----------|------------|
| Identity | memcpy | Contiguous |
| Leading preserved (perm[0]=0) | Copy chunks | Contiguous |
| Trailing preserved | Copy chunks | Contiguous |
| 2D transpose | Cache-blocked | Strided with blocking |
| General | Recursive decomposition | Mixed |

**Implementation:**
```rust
fn permute_tiled<T: Copy>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
) -> Vec<T> {
    let analysis = analyze_permutation(perm);

    match analysis.pattern_type() {
        Pattern::Identity => data.to_vec(),
        Pattern::LeadingPreserved(n) => permute_leading_preserved(data, shape, perm, n),
        Pattern::TrailingPreserved(n) => permute_trailing_preserved(data, shape, perm, n),
        Pattern::Transpose2D => permute_2d_blocked(data, shape, perm),
        Pattern::General => permute_recursive(data, shape, perm),
    }
}
```

### Phase 4: Cache-Blocked 2D Transpose

For the common 2D transpose case (which many high-D permutations reduce to):

```rust
const BLOCK_SIZE: usize = 64; // Tune for L1 cache (typically 32KB)

fn transpose_2d_blocked<T: Copy>(
    src: &[T],
    dst: &mut [T],
    rows: usize,
    cols: usize,
    src_stride: usize,
    dst_stride: usize,
) {
    // Process in BLOCK_SIZE x BLOCK_SIZE tiles
    for col_block in (0..cols).step_by(BLOCK_SIZE) {
        for row_block in (0..rows).step_by(BLOCK_SIZE) {
            // Transpose one tile
            let col_end = (col_block + BLOCK_SIZE).min(cols);
            let row_end = (row_block + BLOCK_SIZE).min(rows);

            for c in col_block..col_end {
                for r in row_block..row_end {
                    dst[c * dst_stride + r] = src[r * src_stride + c];
                }
            }
        }
    }
}
```

### Phase 5: Permutation Decomposition (EITHOT Method)

For complex permutations, decompose into simpler operations:

```rust
/// Decompose permutation into sequence of transpositions
fn decompose_permutation(perm: &[usize]) -> Vec<(usize, usize)> {
    // Use cycle decomposition
    // Each cycle (a b c d) becomes transpositions (a d)(a c)(a b)
    let mut result = Vec::new();
    let mut visited = vec![false; perm.len()];

    for i in 0..perm.len() {
        if visited[i] || perm[i] == i {
            continue;
        }
        // Follow the cycle
        let mut cycle = vec![i];
        let mut j = perm[i];
        while j != i {
            visited[j] = true;
            cycle.push(j);
            j = perm[j];
        }
        // Convert cycle to transpositions
        for k in (1..cycle.len()).rev() {
            result.push((cycle[0], cycle[k]));
        }
    }
    result
}
```

### Phase 6: Recursive Cache-Oblivious Algorithm

```rust
const CACHE_THRESHOLD: usize = 32 * 1024 / std::mem::size_of::<f64>(); // ~4K elements

fn permute_recursive<T: Copy + Default>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
) -> Vec<T> {
    let numel: usize = shape.iter().product();

    // Base case: fits in cache, use simple method
    if numel <= CACHE_THRESHOLD {
        return permute_simple(data, shape, perm);
    }

    // Find largest dimension to split
    let split_dim = shape.iter().enumerate()
        .max_by_key(|(_, &s)| s)
        .map(|(i, _)| i)
        .unwrap();

    // Split tensor in half along split_dim
    let mid = shape[split_dim] / 2;

    // Recursively permute each half
    let (left, right) = split_tensor(data, shape, split_dim, mid);
    let left_result = permute_recursive(&left, &new_shape_left, perm);
    let right_result = permute_recursive(&right, &new_shape_right, perm);

    // Merge results
    merge_tensors(&left_result, &right_result, output_shape, split_dim)
}
```

### Phase 7: SIMD Vectorization

For the innermost loops, use explicit SIMD:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Strided gather for f64
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn gather_f64_avx2(src: *const f64, indices: &[usize; 4]) -> __m256d {
    let idx = _mm256_set_epi64x(
        indices[3] as i64,
        indices[2] as i64,
        indices[1] as i64,
        indices[0] as i64,
    );
    _mm256_i64gather_pd(src, idx, 8)
}
```

### Phase 8: Parallel Outer Loop

For large tensors, parallelize the outer loop:

```rust
#[cfg(feature = "parallel")]
fn permute_parallel<T: Copy + Default + Send + Sync>(
    data: &[T],
    shape: &[usize],
    perm: &[usize],
) -> Vec<T> {
    use rayon::prelude::*;

    let analysis = analyze_permutation(perm);
    let chunk_size = analysis.inner_contiguous_size();
    let num_chunks = numel / chunk_size;

    let mut result = vec![T::default(); numel];

    // Parallel over chunks
    result.par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, dst_chunk)| {
            let src_offset = compute_src_offset(chunk_idx, &analysis);
            // Copy chunk
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(src_offset),
                    dst_chunk.as_mut_ptr(),
                    chunk_size,
                );
            }
        });

    result
}
```

---

## Testing Strategy

1. **Correctness tests**: Compare with naive implementation for all permutation patterns
2. **Performance benchmarks**:
   - Various tensor shapes (small, medium, large)
   - Various permutation patterns (identity, transpose, cyclic, random)
3. **Cache profiling**: Use `perf stat` to measure cache misses

---

## Expected Outcomes

| Optimization | Expected Speedup |
|--------------|------------------|
| Leading dim preservation | 2-5x for applicable cases |
| Cache blocking for 2D | 2-3x |
| Permutation decomposition | 1.5-2x for complex perms |
| SIMD vectorization | 2-4x for inner loops |
| Parallelization | Nx for N cores |

**Target**: Match or exceed Julia's performance (16.4s → ~15s)

---

## References

1. Lyakh, D. (2015). An efficient tensor transpose algorithm for multicore CPU, Intel Xeon Phi, and NVidia Tesla GPU. Computer Physics Communications.
2. Hynninen, A., & Lyakh, D. (2017). cuTT: A High-Performance Tensor Transpose Library for CUDA Compatible GPUs. arXiv:1705.01598.
3. Springer, P., & Bientinesi, P. (2016). TTC: A Tensor Transposition Compiler. arXiv:1603.02297.
