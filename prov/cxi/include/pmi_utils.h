/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2016-2021 Cray Inc. All rights reserved.
 */

/**
 * @defgroup PMI_UTILS_HEADERS pmi_utils.h
 * @ingroup UTILITY_HEADERS
 *
 * @section pmi_utils_header_overview Overview:
 * - PMI Process Management (PM) utility headers for the Libfabric test environment.
 *
 * @section pmi_utils_header_details Details:
 * - pmi_utils.h
 *
 * @see pmi_utils.c
 *
 * @author Joseph Nemeth 10/14/2021.
 * @author based on work by Anthony J. Zinger 01/30/2017.
 *
 * @file pmi_utils.h
 *
 */

#ifndef PMI_UTILS_H
#define PMI_UTILS_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Set the PMI PM debugging level.
 *
 * @param[in] new_debug_level Debug level to use (0 disables)
 *
 * @return void
 */
void pmi_set_debug(int new_debug_level);

/**
 * @brief Terminates the job across all nodes.
 *
 * Abort with an error code of 0 is a normal exit across all nodes.
 *
 * @param[in] error Error code to return
 * @param[in] fmt   printf-style format
 * @param[in] ...   printf-style arguments
 *
 * @return void
 */
void pmi_Abort(int error, const char *fmt, ...);

/**
 * @brief Initializes the pmi utils.
 *
 * This can be called multiple times. The first call terminates the system,
 * other calls are no-ops.
 *
 * @param[out] numranks (optional) returns number of ranks in job
 * @param[out] rank     (optional) returns rank of this instance
 * @param[out] appnum   (optional) returns application number
 *
 * @return void
 */
void pmi_Init(int *numranks, int *rank, int *appnum);

/**
 * @brief Terminates the pmi utils.
 *
 * This can be called multiple times. The first call terminates the system,
 * other calls are no-ops.
 *
 * @return void
 */
void pmi_Finalize(void);

/**
 * @brief Returns the rank of the caller within the job.
 *
 * @param[out] rank The PMI rank of this process.
 *
 * @return void
 */
void pmi_GetRank(int *rank);

/**
 * @brief Returns the number of ranks in the job.
 *
 * @param[out] numranks The number of ranks of this job.
 *
 * @return void
 */
void pmi_GetNumRanks(int *numranks);

/**
 * @brief Returns the number of KVS entries consumed.
 *
 * @param bufsize The size of the source buffer
 * @return int    The number of KVS entries used in the send
 */
int pmi_GetKVSCount(size_t bufsize);

/**
 * @brief Performs a synchronizing Barrier across all jobs.
 *
 * @return void
 */
void pmi_Barrier(void);

/**
 * @brief Broadcasts data from a source rank to all other ranks.
 *
 * @param[in] src_rank The rank supplying the data to broadcast.
 * @param[in] buffer   The value to be broadcast to all of the other ranks.
 * @param[in] len      The length of the value to be broadcast.
 *
 * @return void
 */
void pmi_Bcast(int src_rank, void *buffer, size_t len);

/**
 * @brief Performs an Allgather operation across all ranks.
 *
 * @param[in] srcbuf       The value to send to all of the other ranks.
 * @param[in] len_per_rank The length of the value contributed.
 * @param[out] tgtbuf      An array of src values from all of the ranks.
 *
 * @return void
 */
void pmi_Allgather(void *srcbuf, size_t len_per_rank, void *tgtbuf);

#endif /* PMI_UTILS_H */
