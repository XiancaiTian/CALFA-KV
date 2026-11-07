# CALFA-KV Coordinates and Initialization

This directory implements coordinate construction and discrete optimization within joint optimization. Channel scaling, magnitude storage, grouping selection, and finite-capacity codebook fitting form a single numerical workflow within joint optimization.

## Workflow

| Step | Inputs and operation | Output |
|---|---|---|
| 1 | Fix the model and 64 original windows. Capture K after k_norm and before RoPE, along with V, queries, and signed gradients | Features and codebook-initialization Fisher weights from the same model state |
| 2 | Derive positive channel scaling D from causal query energy. Define features using the stored FP16 magnitude N | Encoding coordinates and a fixed reduction rule |
| 3 | Match complete RoPE pairs using normalized feature covariance and fit G8/256 codebooks | Initial P/C |
| 4 | Run two rounds of finite-rate regrouping. Accept or revert based on calibration error, then retrain K codebooks for changed layers | Grouping verified with the actual finite-capacity codebooks |
| 5 | Search P/C jointly by comparing swap candidates with matched no-swap controls retrained under the same budget | Selected P and its matching K codebooks |
| 6 | Fix D/P and jointly optimize the K/V codebooks under a post-projection attention output objective | Warm-start assets for final continuous calibration |

With D/P fixed, joint calibration starts from μ=0 and A=D and learns μ/A/Cᴷ/Cⱽ together. See [joint optimization](../README.md). All initialization and search operations belong to Part 1.

## Channel, magnitude, and grouping coordinates

We omit the layer index. There are H=8 KV heads, each associated with 4 query heads. The head dimension is d=128, for a total of 1024 channels. D=diag(w) is positive channel scaling, and P is a block channel permutation organized by head. Each 8-dimensional group contains 4 RoPE pairs.

Let e be the pre-RoPE reconstruction error, q̄ the post-RoPE query, and R the rotation matrix at the corresponding position. The logit error is given below.

```math
\Delta\ell_{s,t,a}=\bar q_{s,a}^{\top}R_t e_{t,h}/\sqrt d.
```

Average over query heads sharing the same KV head, across windows and the set T of valid causal position pairs. Take the diagonal of the resulting energy matrix and normalize per head.

```math
M_h=\frac1{4|T|}\sum_{(s,t)\in T}\sum_{a\mapsto h}
R_t^\top\bar q_{s,a}\bar q_{s,a}^\top R_t,\qquad m_{h,j}=(M_h)_{jj}.
```

```math
\epsilon_h=\max(10^{-30},10^{-6}\max_j m_{h,j}),\quad
b_{h,j}=\max(m_{h,j},\epsilon_h),\quad
w_{h,j}=\sqrt{b_{h,j}/\mathrm{mean}_i b_{h,i}}.
```

D changes the error coordinates and remains positive so that the transform is invertible. Once computed, D is stored in FP32 and fixed. Initialization uses μ=0 and A=D.

```math
N_t=\mathrm{FP16}\!\left[\max\!\left(10^{-4},\sqrt{\|Dk_t\|^2/1024}\right)\right],\qquad
z_t=PDk_t/\mathrm{FP32}(N_t).
```

In practice, computing N uses an FP32 pairwise addition tree over adjacent channels in the original head-major order. N is then stored in FP16. Apply P after computing N. Statistics, training, native encoding, and cache migration all use the same stored N. V uses unit scale. See [calfa_scale.py](../../backend/python/calfa_scale.py).

Initial grouping uses the covariance Σ of normalized K accumulated in FP64. Define the following merge cost for a candidate channel set I.

```math
\mathrm{cost}(I)=\det\!\left(\Sigma_h[I,I]+\varepsilon_h I_{|I|}\right)^{1/|I|},\qquad
\varepsilon_h=\max(10^{-30},10^{-4}\mathrm{tr}\Sigma_h/d).
```

Start with each {j,j+64} pair as a complete unit. Use minimum-weight perfect matching to merge 2-dimensional units into 4-dimensional groups and then into 8-dimensional groups. The low and high members of each pair remain adjacent in the permutation to preserve RoPE pairing.

## Finite-capacity codebooks and joint grouping selection

K selects codes in z, and V selects codes within its original contiguous groups. Each group has B=256 codewords and uses Euclidean nearest neighbors. During initialization, squared gradients yield scalar sample weights for the weighted Lloyd update.

```math
a_t=\arg\min_b\|x_t-C_b\|^2,\qquad
C_b\leftarrow\frac{\sum_{t:a_t=b}\omega_t x_t}{\sum_{t:a_t=b}\omega_t}.
```

Codebook-initialization Fisher weights use gradients captured after k_norm and before RoPE for K, and at the output node for V. The N/D compensation uses the fixed stored N.

```math
\omega^K_{t,h,g}=\mathrm{FP32}(N_t)^2
\sum_{j\in I_{h,g}}F^K_{t,h,j}/\max(w_{h,j}^2,10^{-12}),\qquad
\omega^V_{t,h,g}=\sum_{j\in I^V_{h,g}}F^V_{t,h,j}.
```

F is the squared activation gradient of the mean shifted next-token cross-entropy over the full window. These weights are used to fit the initial codebooks. FAKV's Anchor Fisher is calibrated separately after the final assets are frozen.

The covariance cost does not explicitly account for the capacity of 256 codewords, so two rounds of finite-rate search follow. Each candidate repartitions the 8 pairs from two 8-dimensional groups. Count each unordered partition once, giving 35 partitions. Inner fitting and selection and outer calibration use fixed splits. Reconstruction error with the fitted codebooks and the attention checks determine whether to accept or revert. See [config.json](config.json) for the parameters and thresholds.

The subsequent P/C search proposes 6 candidates per head per round. It retrains the swapped groups and the corresponding groups in the matched no-swap control. A candidate must meet the KL improvement threshold against both the current result and the matched control, while satisfying the output and source error bounds. Acceptance retains the exact measured P/C, including its fitted codebooks. See [grouping.py](../core/grouping.py) and [grouping_rules.py](../core/grouping_rules.py).

## Interface to continuous joint calibration

With D/P fixed, warm-start both codebooks using a shared attention output error. The starting codebook RMS for each side, head, and group defines the coordinate unit σ with a floor of 10⁻⁶, and u=C/σ.

```math
\min_{C_K,C_V}\frac1{2E}\sum_{w\in\mathrm{FIT48}}
\|\hat o_w(C_K,C_V)-o_w^\star\|^2,\qquad u=C/\sigma.
```

The next stage uses a similar output calibration objective that includes μ/A. It fixes the hard indices to compute a continuous direction, then decides whether to accept the candidate after actual FP16 re-encoding. See [core/](../core/) for GN/CG, backtracking, and acceptance. Grouping uses the original 2048-token windows. Final continuous calibration uses template-wrapped 2056-token windows. Each stage uses inputs that match its reference states.

## Files and verification

| Path | Purpose |
|---|---|
| [config.json](config.json) | Current recipe and fixed thresholds |
| [finite_rate_groups.py](finite_rate_groups.py), [pair_groups.py](pair_groups.py) | Numerical kernels for the two grouping rounds, pair constraints, and acceptance rules |
| [attention_proxy.py](attention_proxy.py) | Attention checks during grouping |
| [run_group_search.py](run_group_search.py) | Entry point for running a specified grouping pass |
| [initialization/](initialization/) | The zero-center initialization assets for this configuration |
| [inputs/](inputs/) | Fixed original and template-wrapped windows, with an input manifest |

Run the following commands from the CALFA-KV root.

```bash
python3 joint_optimization/coordinates/verify.py --assets
python3 joint_optimization/coordinates/run_group_search.py --help
```

To rebuild the grouping, supply features and queries from a new run along with the current W/P. Full model capture and native codebook retraining are separate steps from the two-round search. The remaining numerical implementations are in `tools/`, `core/`, and the shared `backend/kvquant/`. The bundled initialization and final assets can be reused for the current configuration. Starting from model weights requires fresh captures, a full run, and numerical verification.
