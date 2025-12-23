# Distributed Training Algorithms Using MPI

## Overview
This project simulates and benchmarks different **All-Reduce** strategies used in Distributed Deep Learning (e.g., TensorFlow, PyTorch). It solves the problem of synchronizing gradients across multiple nodes efficiently.



## Algorithms Implemented

### 1. Naive Approach
*   **Structure:** Star topology.
*   **Mechanism:** All nodes send data to Rank 0. Rank 0 sums and sends back.
*   **Bottleneck:** Rank 0 bandwidth. Cost is $O(N \cdot M)$ where N is nodes, M is data size.

### 2. Tree All-Reduce
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

## Configuration
The project uses a `config.txt` file to define simulation constants avoiding "magic numbers".

**config.txt**
```text
MASTER_RANK=0
DATA_TAG=1
```

- `MASTER_RANK`: The MPI rank responsible for coordination (Naive approach) or reporting (Main).
- `DATA_TAG`: The MPI tag used for data communication.

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

## Generating Documentation
The code is fully documented using Doxygen-style comments. You can generate a modern HTML documentation site locally.


### 1. Install Doxygen and Graphviz
Ubuntu/Debian/WSL:
```bash
sudo apt-get install doxygen graphviz
```
macOS:
```bash
brew install doxygen graphviz
```

### 2. Generate HTML
Run the following command in the project root:
```bash
doxygen Doxyfile
```
### 3. View Documentation
Open the generated HTML file in your browser: `./docs/index.html`

## Project Structure

*   **`algorithms.hpp`**: Interface definition for the reduction algorithms.
*   **`algorithms.cpp`**: Implementation logic for Naive and Ring algorithms.
*   **`main.cpp`**: Driver code that handles data initialization, benchmarking, and verification.
*   **`constants.hpp/cpp`**: Logic to read config.txt and expose global settings.
*   **`config.txt`**: Runtime settings.
*   **`Makefile`**: Automated build script.
*   **`Doxyfile`**: Configuration for generating documentation.
*   **`doxygen-awesome.css`**: Custom CSS theme for modern documentation styling.

