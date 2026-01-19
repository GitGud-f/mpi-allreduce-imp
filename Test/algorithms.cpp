/**
 * @file algorithms.cpp
 * @brief Implementation of the All-Reduce algorithms.
 * @details Contains the logic for Naive, Ring, and Tree reduction using MPI.
 */

#include "algorithms.hpp"
#include "constants.hpp"
#include <mpi.h>
#include <vector>
#include <iostream>
using namespace std;

void naive_allreduce(const vector<float>& input, vector<float>& output, int rank, int size){
    output = input;
    int count = input.size();

    if(rank == Config::MASTER_RANK) { // Master
        vector<float> temp_buffer(count);
        for(int j = 0; j < size-1; j++){
            MPI_Status status;

            MPI_Recv(temp_buffer.data(), count, MPI_FLOAT, MPI_ANY_SOURCE, Config::DATA_TAG, MPI_COMM_WORLD, &status);
            for(int i = 0; i < count; i++){
                output[i] += temp_buffer[i];
            }
        }

        for(int dest = 0; dest < size; dest++){
            if(dest == Config::MASTER_RANK)
                continue;
            MPI_Send(output.data(), count, MPI_FLOAT, dest, Config::DATA_TAG, MPI_COMM_WORLD);
        }
    } else{ // Worker

        MPI_Send(input.data(), count, MPI_FLOAT, Config::MASTER_RANK, Config::DATA_TAG, MPI_COMM_WORLD);

        MPI_Recv(output.data(), count, MPI_FLOAT, Config::MASTER_RANK, Config::DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void ring_allreduce(const vector<float>& input, vector<float>& output, int rank, int size){
    int count = input.size();

    if (count % size != 0) {
        if (rank == 0) cerr << "[Error] Ring All-Reduce requires data size to be divisible by MPI Size." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    output = input;
    int chunk_size = count / size;
    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    vector<float> recv_chunk(chunk_size);

     // --- Phase 1: Scatter-Reduce ---
    for (int i = 0; i < size - 1; ++i) {
        int send_idx = (rank - i + size) % size;
        int recv_idx = (rank - i - 1 + size) % size;
        int send_offset = send_idx * chunk_size;
        int recv_offset = recv_idx * chunk_size;

        MPI_Sendrecv(
            &output[send_offset], chunk_size, MPI_FLOAT, right, Config::DATA_TAG,
            recv_chunk.data(),    chunk_size, MPI_FLOAT, left,  Config::DATA_TAG,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        for (int j = 0; j < chunk_size; ++j) {
            output[recv_offset + j] += recv_chunk[j];
        }
    }

    // --- Phase 2: All-Gather ---
    for (int i = 0; i < size - 1; ++i) {
        int send_idx = (rank - i + 1 + size) % size; 
        int recv_idx = (rank - i + size) % size;
        int send_offset = send_idx * chunk_size;
        int recv_offset = recv_idx * chunk_size;

        MPI_Sendrecv(
            &output[send_offset], chunk_size, MPI_FLOAT, right, Config::DATA_TAG,
            recv_chunk.data(),    chunk_size, MPI_FLOAT, left,  Config::DATA_TAG,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        for (int j = 0; j < chunk_size; ++j) {
            output[recv_offset + j] = recv_chunk[j];
        }
    }

}

void tree_allreduce(const vector<float>& input, vector<float>& output, int rank, int size){
    if ((size & (size - 1)) != 0) {
        // Fallback to Naive or MPI implementation if not power of 2
        if(rank == 0) cerr << "[Warning] Tree algo requires Power-of-2 processes. Switching to Naive." << endl;
        naive_allreduce(input, output, rank, size); 
        return;
    }

    int count = input.size();
    output = input; 
    
    std::vector<float> temp_buffer(count);

    // 1. Reduce Phase (Up the tree)
    for (int step = 1; step < size; step *= 2) {
        if (rank % (2 * step) == 0) {
            // Parent
            int source = rank + step;
            if (source < size) {
                MPI_Recv(temp_buffer.data(), count, MPI_FLOAT, source, Config::DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Accumulate
                for (int i = 0; i < count; ++i) {
                    output[i] += temp_buffer[i];
                }
            }
        } else if (rank % (2 * step) == step) {
            // Child
            int dest = rank - step;
            MPI_Send(output.data(), count, MPI_FLOAT, dest, Config::DATA_TAG, MPI_COMM_WORLD);
            break; 
        }
    }

    // 2. Broadcast Phase (Down the tree)
    int start_step = 1;
    while (start_step * 2 < size) start_step *= 2;

    for (int step = start_step; step >= 1; step /= 2) {
        if (rank % (2 * step) == 0) {
            int dest = rank + step;
            if (dest < size) {
                // Parent
                MPI_Send(output.data(), count, MPI_FLOAT, dest, Config::DATA_TAG, MPI_COMM_WORLD);
            }
        } else if (rank % (2 * step) == step) {
            // Child
            int source = rank - step;
            MPI_Recv(output.data(), count, MPI_FLOAT, source, Config::DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}
