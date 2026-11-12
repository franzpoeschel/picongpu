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

Current branch restarts into the same decomposition (no-op).

Constraints:

- PIConGPU assumes one contiguous Cartesian subdomain per rank.
- Compute a new rectangular decomposition and apply it via the patch-aware
  restart. Arbitrary patch-to-rank mappings need a field/guard refactor.
- Supercell granularity is sufficient; below-supercell is out of scope.
- The open issue is where/how the new decomposition is applied (IO plugin vs
  simulation/PMacc control).

Strategies:

- static `--gridDist` (baseline, not adaptive)
- weighted axis cuts / weighted median (simple, contiguous, 1D only)
- recursive coordinate bisection / Zoltan RCB (multi-D, contiguous-ish)
- space-filling curve (locality-preserving, blocks not rectangular)
- graph partitioning: ParMETIS / PT-Scotch / Zoltan (communication-aware, external dependency)
- diffusive neighbor boundary shifting (keeps rectangular layout, slow convergence)
- work stealing (incompatible with fixed field decomposition)
- particle-only migration (incompatible with local field coupling)

Recommended: start with weighted axis cuts/RCB, optionally refined by boundary
shifting; migrate via the existing in-situ checkpoint + patch-aware restart.

- [ ] Define decomposition ownership and a handoff format (offsets/extents or cuts).
- [ ] Define the load metric (particle counts first; then push/deposition and
      guard-exchange cost).
- [ ] Define the trigger (periodic, hysteresis, imbalance threshold).
- [ ] Collect weights and implement the chosen decomposition.
- [ ] Replace the self-restart target with the computed decomposition.
- [ ] Decide field handling (reload vs redistribute).
- [ ] Validate: imbalanced synthetic cases, no-op equivalence, improved imbalance,
      correct restart, no deadlocks.

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
