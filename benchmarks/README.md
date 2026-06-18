# benchmarks
Runs the Halo2 prover benchmark.

```
./benchmarks/run.sh
```

This runs the Fibonacci guest benchmark and writes metrics to `metrics.json`.

To collect a Firefox profiler trace with `samply`:

```
./benchmarks/run.sh --samply
```

This writes `samply_profile/profile.json.gz`, which can be uploaded to Firefox Profiler.

To collect an NVIDIA Nsight Systems profile:

```
./benchmarks/run.sh --nsys
```
