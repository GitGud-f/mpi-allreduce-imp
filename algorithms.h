/** 
 * @file algorithms.h
 * @brief Interface definitions for Distributed Training All-Reduce algorithms.
*/

#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <vector>
using namespace std;

void naive_allreduce(const vector<float>& input, vector<float>& output, int rank, int size);

void ring_allreduce(const vector<float>& input, vector<float>& output, int rank, int size);

#endif
