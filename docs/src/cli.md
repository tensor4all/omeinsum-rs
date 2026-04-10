# CLI Tool

The `omeinsum` command-line tool optimizes contraction order and executes tensor contractions from JSON files, without writing Rust code.

## Installation

```bash
# From the repository root
make cli

# Or directly with Cargo
cargo install --path omeinsum-cli
```

This installs the `omeinsum` binary to `~/.cargo/bin/`.

## Subcommands

### `omeinsum optimize`

Finds an efficient contraction order for an einsum expression.

```bash
omeinsum optimize "ij,jk,kl->il" --sizes "i=2,j=3,k=4,l=5" -o topology.json
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<expression>` | yes | Einsum in NumPy notation, e.g. `"ij,jk->ik"` |
| `--sizes` | yes | Dimension sizes as `label=size` pairs, e.g. `"i=2,j=3,k=4"` |
| `--method` | no | `greedy` (default) or `treesa` |
| `-o, --output` | no | Output file (default: stdout) |
| `--pretty` | no | `true` or `false`; auto-detects TTY when omitted |

The output is a topology JSON that encodes the contraction tree. Pass it to `omeinsum contract` with `-t`.

#### Greedy parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--alpha` | 0.0 | Weight balancing output size vs input size reduction |
| `--temperature` | 0.0 | Stochastic selection temperature; 0 = deterministic |

Example with stochastic greedy:

```bash
omeinsum optimize "ij,jk,kl->il" --sizes "i=10,j=20,k=30,l=40" \
    --method greedy --alpha 0.5 --temperature 0.1
```

#### TreeSA parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--ntrials` | 10 | Number of independent SA trials |
| `--niters` | 50 | Iterations per temperature level |
| `--betas` | `0.01:0.05:15.0` | Inverse temperature schedule as `start:step:stop` |
| `--sc-target` | 20.0 | Space complexity target threshold |
| `--tc-weight` | 1.0 | Time complexity weight in score function |
| `--sc-weight` | 1.0 | Space complexity weight in score function |
| `--rw-weight` | 0.0 | Read-write complexity weight |

The score function is: `score = tc_weight * 2^tc + rw_weight * 2^rw + sc_weight * max(0, 2^sc - 2^sc_target)`.

Example with custom TreeSA:

```bash
omeinsum optimize "ij,jk,kl,lm->im" --sizes "i=10,j=20,k=30,l=40,m=50" \
    --method treesa --ntrials 20 --niters 100 --sc-target 25.0
```

### `omeinsum contract`

Executes a tensor contraction. Requires either a pre-computed topology (`-t`) or an explicit contraction order (`--expr`).

```bash
# Using a topology from optimize
omeinsum contract tensors.json -t topology.json -o result.json

# Using a parenthesized expression
omeinsum contract tensors.json --expr "(ij,jk),kl->il"
```

| Argument | Required | Description |
|----------|----------|-------------|
| `<tensors>` | yes | Path to tensors JSON file |
| `-t, --topology` | one of `-t` or `--expr` | Topology JSON from `optimize` |
| `--expr` | one of `-t` or `--expr` | Parenthesized expression (see below) |
| `-o, --output` | no | Output file (default: stdout) |
| `--pretty` | no | `true` or `false`; auto-detects TTY when omitted |

## Parenthesized Expressions

The `--expr` flag specifies the contraction order using parentheses. Each pair of parentheses denotes one binary contraction step, executed inside-out:

```
(ij,jk),kl->il      # contract ij,jk first, then result with kl
ij,(jk,kl)->il       # contract jk,kl first, then ij with result
((ab,bc),cd),de->ae  # left-to-right chain: ab*bc, then *cd, then *de
```

Tensors are numbered left-to-right in the flat expression. The parenthesization controls which tensors are contracted first, which can significantly affect performance for large networks.

If you don't know the best order, use `omeinsum optimize` to find one automatically.

## JSON Formats

### Input: Tensors File

```json
{
  "dtype": "f64",
  "order": "row_major",
  "tensors": [
    { "shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] },
    { "shape": [3, 2], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] }
  ]
}
```

| Field | Values | Description |
|-------|--------|-------------|
| `dtype` | `f32`, `f64`, `c32`, `c64` | Scalar type. Complex types interleave real/imaginary: `[re, im, re, im, ...]` |
| `order` | `row_major`, `col_major` | Memory layout of `data` arrays. Default: `col_major` |
| `tensors` | array | Each entry has a `shape` and flat `data` array |

> **Note for Python/NumPy users:** NumPy uses row-major (C order) by default.
> If you export data with `tensor.flatten().tolist()`, set `"order": "row_major"`.
> Omitting the `order` field defaults to `col_major` (Fortran/Julia convention),
> which will silently produce wrong results if your data is row-major.

### Output: Result File

```json
{
  "dtype": "f64",
  "order": "row_major",
  "shape": [2, 2],
  "data": [7.0, 10.0, 15.0, 22.0]
}
```

The output uses the same `order` as the input. The `data` array is the flattened result tensor.

### Intermediate: Topology File

Produced by `optimize`, consumed by `contract -t`. You can treat it as a black box. The structure contains the original expression, a label-to-integer mapping, dimension sizes, and a binary contraction tree.

## Example: Full Pipeline

```bash
# 1. Create input tensors (2x3 matrix times 3x2 matrix, row-major)
cat > /tmp/tensors.json << 'EOF'
{
  "dtype": "f64",
  "order": "row_major",
  "tensors": [
    { "shape": [2, 3], "data": [1, 2, 3, 4, 5, 6] },
    { "shape": [3, 2], "data": [1, 2, 3, 4, 5, 6] }
  ]
}
EOF

# 2. Optimize contraction order
omeinsum optimize "ij,jk->ik" --sizes "i=2,j=3,k=2" -o /tmp/topology.json

# 3. Contract using topology
omeinsum contract /tmp/tensors.json -t /tmp/topology.json

# Output: {"dtype":"f64","order":"row_major","shape":[2,2],"data":[22.0,28.0,49.0,64.0]}

# Or contract directly with --expr (no optimize step needed)
omeinsum contract /tmp/tensors.json --expr "(ij,jk)->ik"

# Same output
```
