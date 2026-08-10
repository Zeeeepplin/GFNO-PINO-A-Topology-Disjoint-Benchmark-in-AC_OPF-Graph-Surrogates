# Final framing and consistency audit

This report records the changes requested in
`GFNO-PINO_final_revision_prompt.md`. Approximate line numbers refer to the
revised `manuscript/main.tex`.

## Task 1 - Title

```diff
- What Improves AC-OPF Graph Surrogates Under Line Outages? A
- Topology-Disjoint Benchmark of Physics Fine-Tuning, Topology Conditioning,
- and Spectral Filtering
+ GFNO-PINO: A Topology-Disjoint Benchmark Isolating Physics Fine-Tuning,
+ Topology Conditioning, and Spectral Filtering in AC-OPF Graph Surrogates
```

Location: title block, lines 46-48. There is no separate running-header or
explicit `pdftitle` field; the compiled PDF metadata inherits the updated
LaTeX title.

## Task 2 - Abstract consistency

No edit. Within the first three sentences, lines 63-68 identify the three
choices as unresolved and state: "To test these choices, we construct ... 
(GFNO-PINO)." This already presents GFNO-PINO as the experimental instrument.

## Task 3 - Section 1 roadmap

```diff
- the proposed GFNO-PINO
+ the benchmarked GFNO-PINO architecture
```

Location: Introduction roadmap, line 207.

## Task 4 - Neutral description audit

```diff
- The proposed method instead treats topology as part of the input function
+ The topology-conditioned formulation under test instead treats topology as
+ part of the input function
```

Location: Section 2.2, lines 265-267.

Two additional consistency-only replacements removed the remaining uses of
"proposed" applied to the paper's own components:

```diff
- the proposed graph-spectral architecture trained
+ the evaluated graph-spectral architecture trained
```

Location: Baselines, line 1041.

```diff
- The proposed output layer guarantees
+ The bounded output layer guarantees
```

Location: Limitations, line 1335.

## Task 5 - Section 5.2 opening

```diff
- Table 5 reports the signed reference-cost difference of Eq. (40) as a
- five-seed mean and standard deviation.
+ Across DC-OPF and the five neural models, Table 5 compares the signed
+ reference-cost difference of Eq. (40); neural entries are five-seed means
+ and standard deviations.
```

Location: Economic performance, lines 1173-1176. No value or subsequent
GFNO-PINO sentence was changed.

## Task 6 - Research gap

No edit. The subsection already treats accuracy as an open empirical question
and states that the benefit is not confirmed under the present protocol.

## Task 7 - Flagged, not auto-edited sentences

None. The full-document scan, excluding the specified Sections 5.3, 5.5, 6,
and 7, found no unhedged success statement about GFNO-PINO, topology
conditioning, or Chebyshev filtering. Statements about cited DeepOPF/PINN work
refer to other authors or the literature and were left unchanged.

## Task 8 - Cover letter

The factual cover letter is 186 words from salutation through the final
technical paragraph, excluding the signature. Source:
`manuscript/cover_letter.tex`.

## Integrity confirmation

No equation, numerical value, table, table caption, figure, figure caption, or
citation was altered in this pass.
