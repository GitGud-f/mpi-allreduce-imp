/**
 * @file main.cpp
 * @brief Entry point for the program
 */

#include "algorithms.h"
#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

#define MASTER_RANK 0


void run_benchmark(const char* name, int rank, function<void(void)> func){
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    func();
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == MASTER_RANK) {
        cout << name << " Time: " << (end - start) * 1000.0 << " ms" << endl;
    }
}
int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 100000*size;
    vector<float> data(N, 1.0f);
    vector<float> result_naive(N), result_native(N);

    if(rank == MASTER_RANK){
        cout << "------------------------------------------------\n";
        cout << "Distributed Training Benchmark (Processes: " << size << ")\n";
        cout << "------------------------------------------------\n";
    }

    run_benchmark("1. Naive Implementation", rank, [&]() {
        naive_allreduce(data, result_naive, rank, size);
    });

    run_benchmark("2. MPI Library (Ref)   ", rank, [&]() {
        MPI_Allreduce(data.data(), result_native.data(), N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    });

    if(rank  == MASTER_RANK){
        float expected = (float)size;
        bool correct = (abs(result_naive[0] - expected) < 1e-5) &&
                       (abs(result_native[0] - expected) < 1e-5);

        cout << "------------------------------------------------\n";
        cout << "Correctness Check: " << (correct ? "PASSED" : "FAILED") << "\n";
        cout << "------------------------------------------------\n";

        cout.flush(); 
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}