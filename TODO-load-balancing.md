# Load balancing: TODO

Scope: streaming / in-situ workflows only; file-based checkpoints are out of scope.

Idea (PR #5405): halt, write a *virtual* checkpoint via streaming, restart from
it with a new decomposition. In-situ restart and load balancing are orthogonal.
The changed-decomposition restart is largely merged; the rest is a prototype.

## 1. Deduplicate output granularity

Three near-duplicate strategies:

- per rank: `writeParticlesWithoutAnySubdividing` + `writeParticlePatches`
- per supercell: `createAChunkForEverySupercell` + `writeParticlesBySupercells` + `writeParticlePatchesBySupercells`
- size-bounded: `createSupercellRangeChunks` + `writeParticlesByChunks` + `writeParticlePatches`

- [ ] Parameterize by `PatchGranularity { PerRank, PerSupercell, SizeBounded }`.
- [ ] Merge chunk construction, particle writing, and patch metadata writing into
      one implementation each; keep strategy-specific validation separate.
- [ ] Do not force identical patch layouts; they are the point. Check
      per-strategy invariants: patch count/layout, `numParticles`/offsets,
      offset/extent coverage, and identical particles after restart.
- [ ] Make the strategy configurable and test all three, including empty
      supercells, unequal local supercell counts, and over-sized supercells.
- [ ] Verify `LoadSpecies` consumes all strategies.

## 2. Properly wire on-the-fly restart

Currently `Checkpoint::checkpoint()` abuses `--checkpoint.period`: under
`MPI_THREAD_MULTIPLE` it writes a checkpoint and asynchronously restarts from it.

- [ ] Separate checkpointing from checkpoint-and-reload.
- [ ] Define stop-the-world semantics: the simulation continues only after the
      reload completes; concurrent write/read is only I/O overlap.
- [ ] Add a dedicated operation/plugin/backend API (`checkpointAndReload()` or a
      `LoadBalance` controller) that owns the writer/reader sync state machine.
- [ ] Add an explicit trigger/CLI; regular checkpointing must never restart.
- [ ] Define backend capabilities (concurrent read-while-write vs not) and fail
      clearly.
- [ ] Do not reload `idProvider`/RNG state during a rebalance; keep the running
      simulation's IDs and RNG stream.
- [ ] Decide handling of PML fields and other plugins with restart state.
- [ ] Keep/verify error propagation; no hangs on reader or writer failure.
- [ ] Investigate relaxing `MPI_THREAD_MULTIPLE`, or at least control SST
      startup more finely.
- [ ] Test: checkpoint-only does not reload; explicit rebalance reloads once;
      failures abort cleanly.

## 3. Actual load balancing

Goal: one-shot, stop-the-world re-decomposition. Recompute the field
decomposition so that the particle subdecomposition is reasonably balanced,
apply it, then continue as if load balancing did not exist. No persistent load
balancing scheduler and no change to the core particle/field algorithm.

Constraints:

- Runtime re-decomposition is the feature to implement, not a blocker.
- Particles must stay with the GPU that owns their field region. Particle-only
  migration or work stealing is out of scope.
- Decomposition must stay a contiguous Cartesian grid. Compute new block
  offsets/extents; arbitrary patch-to-rank mappings are out of scope.
- Moving window: only the slide axis must remain equidistant. Keep that axis
  uniform and restrict rebalancing to the other axes.
- Patch output must tolerate unequal local supercell counts; fix the
  per-supercell writer (or use per-rank patch metadata).
- Supercell granularity is sufficient; below-supercell is out of scope.
- Concurrent checkpoint/restart I/O is an implementation detail; the
  balancing operation itself is stop-the-world.

Candidate decomposition algorithms:

- weighted axis cuts / weighted median (simple, contiguous, 1D only)
- recursive coordinate bisection / Zoltan RCB (multi-D, contiguous-ish)
- space-filling curve (locality-preserving, blocks not rectangular)
- graph partitioning: ParMETIS / PT-Scotch / Zoltan (communication-aware,
  external dependency)
- diffusive neighbor boundary shifting (keeps rectangular layout, slow
  convergence)
- static `--gridDist` is the baseline only, not adaptive

Recommended: start with weighted axis cuts/RCB on the non-slide axes,
optionally refined by boundary shifting. Use the in-situ checkpoint/restart as
the data movement mechanism, with stop-the-world semantics.

- [ ] Implement runtime re-decomposition: update `GridController`,
      `CommunicatorMPI`, `SubGrid`, field and particle buffers, guard exchange,
      mapping descriptions, and plugin state for changed local sizes.
- [ ] Keep the moving-window axis uniform; rebalance only the other axes.
- [ ] Fix per-supercell patch output for unequal local supercell counts, or
      select per-rank patch metadata when sizes differ.
- [ ] Define decomposition ownership and a handoff format (offsets/extents or
      cuts).
- [ ] Define the load metric (particle counts first; then push/deposition and
      guard-exchange cost).
- [ ] Define the stop-the-world trigger (explicit signal or imbalance
      threshold; no continuous scheduler).
- [ ] Collect weights and implement the chosen decomposition.
- [ ] Replace the self-restart target with the computed decomposition.
- [ ] Validate: imbalanced synthetic cases, no-op equivalence, improved
      imbalance, correct restart, no deadlocks.

## 4. Scaling follow-ups (PR #5405)

- [ ] Solve the N->N particle-patch communication in streaming. Options:
      aggregate/deaggregate, ADIOS2 Local Values, or write patches to a parallel
      filesystem (out of scope here).
- [ ] Add a two-level patch hierarchy: per-supercell lower level; per-process
      higher level pointing into it; communicate only the small higher level
      (Local Values).
- [ ] Batch small I/O operations while keeping patches as the logical granularity.
- [ ] Budget SST host-side buffers (roughly a CPU clone of the GPU volume).
      Investigate using node SSDs as SST buffer; until then leave GPU headroom.
- [ ] Decide how PML fields and other stateful plugins participate.

## 5. Working SST configurations

- fabric: `FABRIC_PROVIDER=shm FABRIC_IFACE=shm`, `DataTransport=fabric`
- UCX: `UCX_TLS=shm UCX_POSIX_USE_PROC_LINK=n`, `DataTransport=ucx`
- verified here: `DataTransport=MPI` (least environment-dependent on one node)

All with `PIC_USE_THREADED_MPI=MPI_THREAD_MULTIPLE`, `--checkpoint.openPMD.ext sst`,
and the JSON under `--checkpoint.openPMD.backendConfig` /
`--checkpoint.openPMD.backendConfigRestart`.
