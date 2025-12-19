# Distributed Training Algorithms Using MPI

## Overview
This project simulates and benchmarks different **All-Reduce** strategies used in Distributed Deep Learning (e.g., TensorFlow, PyTorch). It solves the problem of synchronizing gradients across multiple nodes efficiently.

This implementation compares three approaches using **C++** and **MPI**:
1.  **Naive Approach:** Centralized Parameter Server style (Bottlenecked).
2.  **Ring All-Reduce:** Bandwidth-optimal distributed approach.
3.  **MPI Native:** The highly optimized `MPI_Allreduce` provided by the library.

## Project Structure


*   **`algorithms.h`**: Interface definition for the reduction algorithms.
*   **`algorithms.cpp`**: Implementation logic for Naive and Ring algorithms.
*   **`main.cpp`**: Driver code that handles data initialization, benchmarking, and verification.
*   **`Makefile`**: Automated build script.

## Algorithms Implemented

### 1. Naive Approach
*   **Structure:** Star topology.
*   **Mechanism:** All nodes send data to Rank 0. Rank 0 sums and sends back.
*   **Bottleneck:** Rank 0 bandwidth. Cost is $O(N \cdot M)$ where N is nodes, M is data size.

### 2. Tree All-Reduce (Added)
*   **Structure:** Binary/Binomial Tree.
*   **Mechanism:** 
    1.  **Reduce:** Leaves send to parents recursively up to Root (Rank 0).
    2.  **Broadcast:** Root sends final sum down to leaves.
*   **Advantage:** **Latency Optimal**. Takes $O(\log N)$ steps. Good for high latency networks or small message sizes.
*   **Disadvantage:** Not bandwidth optimal for large weights compared to Ring.

### 3. Ring All-Reduce
*   **Structure:** Logical Ring.
*   **Mechanism:** Data is split into chunks. Chunks rotate through the ring (Scatter-Reduce then All-Gather).
*   **Advantage:** **Bandwidth Optimal**. Network usage is constant per node regardless of cluster size. Best for large Neural Networks.


## Prerequisites
*   **C++ Compiler** (g++ or clang)
*   **MPI Implementation** (OpenMPI or MPICH)
*   **Make**

## Building and Running

### 1. Build the project
Run the default make command to compile the source files.
```bash
make
```

### 2. Run the benchmark

By default, the simulation runs with 4 processes.
```bash
make run
```
To run with a different number of processes (e.g., 8):
```bash
make run NP=8
```

### 3. Clean Build
To remove object files and the executable:
```bash
make clean
```
