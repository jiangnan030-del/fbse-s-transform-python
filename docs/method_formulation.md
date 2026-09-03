# FBSE-ST method formulation notes

This note documents the mathematical interpretation currently used in the repository. It is intended to make the implementation easier to audit and refine.

## 1. Signal and Bessel basis

Let the discrete input signal be

\[
x[n], \quad n = 1,2,\dots,N.
\]

Let \(\alpha_m\) denote the positive zeros of the zero-order Bessel function \(J_0\):

\[
J_0(\alpha_m) = 0, \quad m = 1,2,\dots,M.
\]

The discrete Bessel basis used in the code is

\[
D_{m,n} = J_0\!\left(\frac{\alpha_m}{N} n\right),
\quad m = 1,\dots,M,\; n = 1,\dots,N.
\]

In code, this corresponds to the basis matrix built in `build_bessel_basis()`.

## 2. FBSE coefficients

The repository uses the coefficient form

\[
a_m = \frac{2}{N^2 J_1^2(\alpha_m)}
\sum_{n=1}^{N} n\, x[n] \, J_0\!\left(\frac{\alpha_m}{N} n\right).
\]

The coefficient vector is therefore

\[
\mathbf{a} = [a_1, a_2, \dots, a_M]^\top.
\]

This is implemented in `compute_fbse_coefficients()`.

## 3. Bessel pseudo-frequency axis

Because the code operates on a discrete Bessel index domain rather than a standard Fourier frequency grid, the current implementation introduces a pseudo-frequency axis:

\[
f_m \propto \alpha_m.
\]

Optionally, the axis is scaled by signal length and normalized to the interval \((0,1]\):

\[
\tilde f_m = \frac{\alpha_m / N}{\max_k (\alpha_k / N)}.
\]

This is implemented in `compute_bessel_pseudo_frequencies()`.

## 4. Adaptive Gaussian widths

To mimic the S-transform principle that low frequencies use wider windows and high frequencies use narrower windows, the implementation defines:

\[
\sigma_m = \frac{c}{\max(\tilde f_m, \varepsilon)},
\]

where:

- \(c\) is the user parameter `sigma_scale`
- \(\varepsilon\) is a small numerical floor

This is implemented in `compute_adaptive_window_widths()`.

## 5. Pairwise frequency distance matrix

The pairwise Bessel-frequency separation is

\[
\Delta_{m,k} = \tilde f_k - \tilde f_m.
\]

This forms the distance matrix used in the Gaussian weighting operator.

## 6. Gaussian weighting matrix

The code constructs a row-wise adaptive Gaussian weighting matrix:

\[
G_{m,k} = \frac{1}{\sqrt{2\pi}\,\sigma_m}
\exp\!\left(-\frac{\Delta_{m,k}^2}{2\sigma_m^2}\right).
\]

If row normalization is enabled, it applies

\[
G_{m,k} \leftarrow \frac{G_{m,k}}{\sum_j G_{m,j}}.
\]

This normalization makes each row behave like a localized redistribution operator rather than an uncontrolled amplification factor.

This is implemented in `build_frequency_adaptive_gaussian_window()`.

## 7. Toeplitz coefficient matrix

The repository currently supports two Toeplitz constructions.

### 7.1 Page-style mode

This mode follows the wording that the Toeplitz matrix is generated from the FBSE coefficient vector as the first column:

\[
T = \operatorname{Toeplitz}(\mathbf{a}, [a_1, 0, 0, \dots, 0]).
\]

This is the default mode in the current implementation.

### 7.2 Hermitian mode

An alternative experimental mode is:

\[
T = \operatorname{Toeplitz}(\mathbf{a}, \mathbf{a}^H).
\]

This is convenient for some complex-valued exploratory studies but is less directly tied to the page wording.

These are implemented in `build_toeplitz_from_coefficients()`.

## 8. Coefficient-domain weighting

The weighted coefficient-domain matrix is defined elementwise by

\[
W = T \odot G,
\]

where \(\odot\) denotes the Hadamard product.

This is implemented in `apply_fbse_frequency_weighting()`.

## 9. Time-frequency reconstruction

The current reconstruction step uses the Bessel basis matrix directly:

\[
Z = W D,
\]

where:

- \(Z\) is the complex time-frequency matrix
- \(D\) is the Bessel basis matrix

The repository then reports:

\[
|Z|,
\quad
E = |Z|^2.
\]

These are returned respectively as:

- `complex_time_frequency_matrix`
- `time_frequency_matrix`
- `energy_time_frequency_matrix`

This is implemented in `reconstruct_time_frequency()`.

## 10. Interpretation

The current implementation should be understood as a structured computational realization of the page's four-step method description, not yet as a finalized one-to-one transcription of a published derivation.

Its strengths are:

- clear mapping from algorithm steps to code
- explicit control of adaptive widths
- easy experimentation with weighting behavior
- direct comparison with STFT outputs

Its remaining refinement directions include:

- confirming the exact continuous-to-discrete derivation used in the target source
- validating whether the present reconstruction should use a direct basis map, transposed basis, or a normalized inverse operator
- refining normalization constants for coefficient-domain and reconstruction-domain consistency
- benchmarking concentration performance quantitatively
