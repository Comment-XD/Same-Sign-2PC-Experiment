# CrypTen Research — Quick README

This repository contains a small research prototype exploring additive secret sharing, Beaver triples, and secure convolution/linear operations (inspired by CrypTen). The code is minimal and intended for experimentation and algorithmic validation.

## High-level pipeline
1. Prepare plaintext tensors (images, weights).
2. Encode values to fixed-point representation (see `src/encoder.py`).
3. Generate additive secret shares for parties (`src/rng.py`).
4. Run secure primitives (Beaver triple based multiplication, conv, linear) implemented in `src/beaver.py`, `src/conv2d.py`, and `src/linear.py`.
5. Reconstruct and decode results to verify correctness.

## Repo structure
- `main.py` — experiment driver (trials, logging).
- `src/encoder.py` — fixed-point encoder/decoder (handles signed values and overflow).
- `src/rng.py` — random/share generation utilities (additive shares, masks).
- `src/additive_tensor.py` — wrapper class for additive secret tensors (share management).
- `src/beaver.py` — Beaver triple protocol helpers and plain/batched convolution helpers.
- `src/conv2d.py` — Conv2D module using shared tensors / Beaver protocol.
- `src/linear.py` — Linear layer using shared tensors / Beaver protocol.
- `src/module.py` — common module utilities.
- `logs/` — output logs created by `main.py`.

## Data shapes & conventions
- Channel-first format is used: `(batch, channels, height, width)` for batched inputs.
- Kernel weights: `(out_channels, in_channels, kH, kW)`.
- Encoded tensors are integers; decoding returns floats.
- Default ring modulus used in code: `2**64` (implementations must handle signedness carefully — see notes below).

## Important implementation notes
- All arithmetic in secure protocols must be done modulo the ring size to avoid overflows. Use explicit `% modulus` after ops.
- For signed values, map ring integers to signed representation on decode (e.g., values >= ring//2 interpret as negative via subtraction).
- Beaver triples are used to securely multiply shared values. The protocol requires revealing masked differences (ε, δ) and then combining triple shares.
- When using 64-bit rings with signed values, prefer a ring and casting strategy consistent with NumPy/Python integer limits (see `src/encoder.py` and `src/rng.py`).

## Running experiments
- Ensure Python dependencies are installed (NumPy, PyTorch if used).
- Run the main trial:
  ```bash
  python main.py
  ```
- Logs are written to `logs/` by the driver. Check the log file for per-trial summaries.

## Logging
- `main.py` creates timestamped log files in `logs/` (summary statistics per run).
- Enable verbose mode in `trial()` to log intermediate results.

## Tests & validation
- Small unit tests: compare plaintext convolution/linear results with reconstructed shared results.
- Test with small bit-lengths (e.g., 8 or 16 bits) before scaling up to 64-bit rings to avoid overflow while debugging.

## Extension ideas
- Vectorize convolution and Beaver operations with NumPy broadcasting for speed.
- Replace local RNG-based shares with PRG-based PRSS/PRZS if simulating multiple parties.
- Add secure comparison (binary sharing) and ReLU using multiplexing.

## References
- Beaver triples for secure multiplication
- Fixed-point encodings and modular arithmetic in MPC
- CrypTen implementation patterns

If anything in the pipeline behaves unexpectedly (encoding errors, overflow warnings), inspect `src/encoder.py` and `src/rng.py` first — those usually explain most issues with precision and wrapping.