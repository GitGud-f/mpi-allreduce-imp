/**
 * @file algorithms.cpp
 * @brief Implementation of the All-Reduce algorithms.
 */

#include "algorithms.h"
#include <mpi.h>
#include <vector>
using namespace std;

// MPI Tag for Sending Data
#define DATA_TAG 1
// Rank of the master process
#define MASTER_RANK 0

void naive_allreduce(const vector<float>& input, vector<float>& output, int rank, int size){
    output = input;
    int count = input.size();

    if(rank == MASTER_RANK) { // Master
        vector<float> temp_buffer(count);
        for(int src = 1; src < size; src++){
            MPI_Recv(temp_buffer.data(), count, MPI_FLOAT, src, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int i = 0; i < count; i++){
                output[i] += temp_buffer[i];
            }
        }

        for(int dest = 1; dest < size; dest++){
            MPI_Send(output.data(), count, MPI_FLOAT, dest, DATA_TAG, MPI_COMM_WORLD);
        }
    } else{ // Worker

        MPI_Send(input.data(), count, MPI_FLOAT, MASTER_RANK, DATA_TAG, MPI_COMM_WORLD);

        MPI_Recv(output.data(), count, MPI_FLOAT, MASTER_RANK, DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void ring_allreduce(const vector<float>& input, vector<float>& output, int rank, int size){
    output = input;
    int count = input.size();

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
            &output[send_offset], chunk_size, MPI_FLOAT, right, DATA_TAG,
            recv_chunk.data(),    chunk_size, MPI_FLOAT, left,  DATA_TAG,
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
            &output[send_offset], chunk_size, MPI_FLOAT, right, DATA_TAG,
            recv_chunk.data(),    chunk_size, MPI_FLOAT, left,  DATA_TAG,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE
        );

        for (int j = 0; j < chunk_size; ++j) {
            output[recv_offset + j] = recv_chunk[j];
        }
    }

}
