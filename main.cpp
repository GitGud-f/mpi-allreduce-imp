/**
 * @file main.cpp
 * @brief Entry point for the Distributed Training Simulation.
 * @details Initialized MPI, loads configuration, generates data,
 *          runs benchmarks, and validates results.
 */

#include "algorithms.hpp"
#include "constants.hpp"
#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <functional>
using namespace std;


/**
 * @brief Runs a specific algorithm and measures execution time.
 * 
 * @param name Name of the algorithm for display.
 * @param rank Current MPI rank.
 * @param func Lambda function containing the algorithm call.
 */
void run_benchmark(const char* name, int rank, function<void(void)> func){
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    func();
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == Config::MASTER_RANK) {
        cout << name << " Time: " << (end - start) * 1000.0 << " ms" << endl;
    }
}
int main(int argc, char** argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Config::load("config.txt");

    const int MIN_N = 1000000;
    const int N = (MIN_N / size) * size; // to ensure divisibility
    
    vector<float> data(N, 1.0f);
    vector<float> result_naive(N), result_native(N), result_tree(N), result_ring(N);

    if(rank == Config::MASTER_RANK){
        cout << "------------------------------------------------\n";
        cout << "Distributed Training Benchmark (Processes: " << size << ")\n";
        cout << "------------------------------------------------\n";
    }

    run_benchmark("1. Naive Implementation", rank, [&]() {
        naive_allreduce(data, result_naive, rank, size);
    });

    run_benchmark("2. Ring Implementation ", rank, [&]() {
        ring_allreduce(data, result_ring, rank, size);
    });

     run_benchmark("3. Tree Implementation ", rank, [&]() {
        tree_allreduce(data, result_tree, rank, size);
    });

    run_benchmark("4. MPI Library (Ref)   ", rank, [&]() {
        MPI_Allreduce(data.data(), result_native.data(), N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    });

    if(rank  == Config::MASTER_RANK){
        float expected = (float)size;
        bool correct = (abs(result_naive[0] - expected) < 1e-5) &&
                       (abs(result_ring[0] - expected) < 1e-5) &&
                       (abs(result_tree[0] - expected) < 1e-5) &&
                       (abs(result_native[0] - expected) < 1e-5);

        cout << "------------------------------------------------\n";
        cout << "Correctness Check: " << (correct ? "PASSED" : "FAILED") << "\n";
        cout << "------------------------------------------------\n";

        cout<<endl; 
    }

    MPI_Finalize();
    return 0;
}