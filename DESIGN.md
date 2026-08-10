# Graph-spectral design

The default operator uses a Chebyshev polynomial parameterization of a spectral
graph filter:

\[
h'=\sum_{k=0}^{K-1}T_k(\tilde L)hW_k,\qquad
T_k(x)=2xT_{k-1}(x)-T_{k-2}(x).
\]

`L` is the normalized, topology-conditioned graph Laplacian and
`\tilde L=L-I`, using the standard bound `lambda_max <= 2`. Each line outage
therefore changes the operator *input* and its message-passing paths, without
changing learned parameters.

## Why Chebyshev is the publication default

- It is basis-free. Laplacian eigenvectors can rotate, flip sign, or exchange
  within repeated eigenspaces after a single outage. Direct graph-Fourier
  coefficients are consequently not aligned across topologies without an
  additional basis-matching method.
- It needs no eigendecomposition per contingency and costs
  `O(K |E| C_in C_out)` with sparse edges (the implementation also accepts
  dense padded Laplacians for simple batching).
- A `K`-term polynomial is a genuine localized spectral multiplier, retaining
  the spectral-operator inductive bias needed for the GFNO ablation.

An eigenbasis implementation can represent unrestricted global spectral
multipliers with fewer layers on a fixed graph. Its `O(N^3)` preprocessing,
`O(N^2)` transforms, and topology-dependent basis ambiguity make it a poor
default for unseen N-1 grids. It remains a worthwhile fixed-topology baseline,
but is not used by the topology-conditioned model.

Hard box constraints are applied to voltage magnitude and generator active and
reactive power. They do **not** guarantee simultaneous AC equality feasibility
or line feasibility. Those coupled, non-convex constraints are optimized using
physics residuals and are reported explicitly at evaluation time. Predictor
edges retain all four directional two-port coefficients
`Yff, Yft, Ytt, Ytf`; this is essential for transformer taps and phase shifts.
The canonical edge axis is the union of physical lines and fixed transformers.
The full status map uses each physical-line contingency status and assigns
status one to transformers. Both learned edge messages and the normalized
Laplacian use this status, and endpoint aggregation divides by active
post-outage degree.
Here `K` denotes the number of Chebyshev terms, so the polynomial degree and
maximum propagation radius are `K - 1`.
