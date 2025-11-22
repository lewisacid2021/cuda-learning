# Introduction

**Some simple samples in CUDA learning.**

- Use build.sh to build all samples.
```
    cd scripts/
    ./build.sh 
```

- Use run.sh to run all samples or specific sample with -case <name>.
```
    cd scripts/
    ./run.sh    // run all samples
    ./run.sh -case hello_world // run hello_world
```

- Use perf.sh to run with ncu to generate performance report.
```
    cd scripts/
    ./perf.sh    // run all samples
    ./perf.sh -case hello_world // run hello_world
```

- Use clean.sh to clean all build cache and executables.
```
    cd scripts/
    ./clean.sh
```