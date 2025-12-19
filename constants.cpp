/**
 * @file constants.cpp
 * @brief Implementation of the configuration loader.
 */

#include "constants.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <algorithm>

using namespace std;

namespace Config {
    // Default values (fallback)
    int MASTER_RANK = 0;
    int DATA_TAG = 1;

    void load(const string& filename) {
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "[Warning] Could not open " << filename 
                      << ". Using defaults (Master: 0, Data Tag: 1)." << endl;
            return;
        }

        string line;
        while (getline(file, line)) {

            line.erase(remove(line.begin(), line.end(), ' '), line.end());
            
            // Skip comments or empty lines
            if (line.empty() || line[0] == '#') continue;

            size_t delimiterPos = line.find('=');
            if (delimiterPos != string::npos) {
                string key = line.substr(0, delimiterPos);
                string value = line.substr(delimiterPos + 1);

                try {
                    if (key == "MASTER_RANK") {
                        MASTER_RANK = stoi(value);
                    } else if (key == "DATA_TAG") {
                        DATA_TAG = stoi(value);
                    }
                } catch (...) {
                    cerr << "[Error] Invalid number format in config for key: " << key << std::endl;
                }
            }
        }
        file.close();
    }
}