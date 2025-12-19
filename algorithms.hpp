/** 
 * @file algorithms.h
 * @brief Interface definitions for Distributed Training All-Reduce algorithms.
 * @details Contains declarations for Naive, Ring, and Tree-based reduction strategies.
*/

#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <vector>
using namespace std;

/**
 * @brief Naive All-Reduce Strategy (Centralized Parameter Server).
 * @details All workers send gradients to the Master node. The Master aggregates
 *          them and broadcasts the result back. High bandwidth bottleneck at Master.
 * 
 * @param input  Local data vector (gradients) from this process.
 * @param output Vector to store the final reduced results.
 * @param rank   Current MPI process rank.
 * @param size   Total number of MPI processes.
 */
void naive_allreduce(const vector<float>& input, vector<float>& output, int rank, int size);

/**
 * @brief Ring All-Reduce Strategy (Bandwidth Optimal).
 * @details Nodes are arranged in a logical ring. Data is split into chunks.
 *          Performs a Scatter-Reduce phase followed by an All-Gather phase.
 *          Constant bandwidth usage per node regardless of cluster size.
 * 
 * @param input  Local data vector.
 * @param output Vector to store the result.
 * @param rank   Current MPI process rank.
 * @param size   Total number of MPI processes.
 */
void ring_allreduce(const vector<float>& input, vector<float>& output, int rank, int size);

/**
 * @brief Tree All-Reduce Strategy (Latency Optimal).
 * @details Uses a recursive binary/binomial tree structure.
 *          Phase 1: Recursive doubling reduction to Root.
 *          Phase 2: Recursive halving broadcast from Root.
 *          Performs in O(log N) steps.
 * 
 * @param input  Local data vector.
 * @param output Vector to store the result.
 * @param rank   Current MPI process rank.
 * @param size   Total number of MPI processes.
 */
void tree_allreduce(const vector<float>& input, vector<float>& output, int rank, int size);

#endif // ALGORITHMS_H
