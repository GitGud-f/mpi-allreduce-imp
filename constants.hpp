/**
 * @file constants.hpp
 * @brief Global configuration definitions.
 * @details Defines the namespace for global settings which are loaded 
 *          from an external configuration file.
 */

#pragma once
#include <string>

namespace Config {
    /** @brief The MPI Rank ID considered as the Master node. */
    extern int MASTER_RANK;

    /** @brief The MPI Tag used to identify data messages. */
    extern int DATA_TAG;

    /**
     * @brief Loads configuration settings from a file.
     * @param filename Path to the configuration file (e.g., "config.txt").
     */
    void load(const std::string& filename);
}