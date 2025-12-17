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
