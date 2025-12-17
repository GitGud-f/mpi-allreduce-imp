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

## Theory

### 1. Naive Approach
*   **Mechanism:** All workers send gradients to Rank 0. Rank 0 sums them and broadcasts back.
*   **Drawback:** Rank 0 becomes a bottleneck. Communication cost scales linearly with $N$ (number of nodes).

### 2. Ring All-Reduce
*   **Mechanism:** Nodes are arranged in a logical ring.
    *   *Scatter-Reduce Phase:* Data chunks circulate the ring; each node accumulates a specific chunk.
    *   *All-Gather Phase:* Fully accumulated chunks circulate the ring until every node has the full data.
*   **Advantage:** Bandwidth optimal. Communication cost is constant regardless of $N$ (in terms of bandwidth usage per link).

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
