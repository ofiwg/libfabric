/*
 * Copyright (c) 2016-2017 Cray Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @defgroup LTU_SUPPORT_PM_HEADERS ltu_utils_pm.h
 * @ingroup UTILITY_HEADERS
 *
 * @section ltu_utils_pm_header_overview Overview:
 * - LTU Process Management (PM) utility headers for the Libfabric test environment.
 *
 * @section ltu_utils_pm_header_details Details:
 * - ltu_utils_pm.h
 *
 * @see ltu_utils_pm.c
 *
 * @author Anthony J. Zinger 01/30/2017.
 * @file ltu_utils_pm.h
 *
 */

#ifndef LTU_UTILS_PM_H
#define LTU_UTILS_PM_H

#include <stdint.h>
#include <stddef.h>

/**
 * ltu_get_pm_version() - LTU utility function to return an integer value for the version of PM
 *
 * @brief Return an integer value for the version of PM.
 *
 * @return string - containing the version of PM.
 */
char * ltu_get_pm_version(void);

/* ltu_utils_pm.c out-of-band Process Management stuff (i.e., PMI) */
/**
 * ltu_pm_Init() - LTU utility function to wrap the PMI Init API.
 *
 * @brief Wrapper for the PMI Init API.
 *
 * @param[in] argc The application's argc parameter.
 * @param[in] argv The application's argv parameter.
 *
 * @return void
 */
void ltu_pm_Init(int *, char ***);

/**
 * ltu_pm_Finalize() - LTU utility function to wrap the PMI Finalize API.
 *
 * @brief Wrapper for the PMI Finalize API.
 *
 * @return void
 */
void ltu_pm_Finalize(void);

/**
 * ltu_pm_Abort() - LTU utility function to wrap the PMI Abort API.
 *
 * @brief Wrapper for the PMI Abort API.
 *
 * @return void
 */
void ltu_pm_Abort(void);

/**
 * ltu_pm_Exit() - LTU utility function to wrap the PMI Exit API.
 *
 * @brief Wrapper for the PMI Exit API.
 *
 * @return void
 */
void ltu_pm_Exit(void);

/**
 * ltu_pm_Exit() - LTU utility function to wrap the PMI Exit API.
 *
 * @brief Wrapper for the PMI Exit API.
 *
 * @param[in] code The return code that this job will exit with.
 *
 * @return void
 */
void ltu_pm_Exit_with_code(int code);

/**
 * ltu_pm_Rank() - LTU utility function to wrap the PMI Get_Rank API.
 *
 * @brief Wrapper for the PMI Get_Rank API.
 *
 * @param[out] rank The PMI rank of this process.
 *
 * @return void
 */
void ltu_pm_Rank(int *);

/**
 * ltu_pm_Job_size() - LTU utility function to wrap the PMI Get_Size API.
 *
 * @brief Wrapper for the PMI Get_Size API.
 *
 * @param[out] nranks The number of ranks of this job.
 *
 * @return void
 */
void ltu_pm_Job_size(int *);

/**
 * ltu_pm_Clique_size() - LTU utility function to wrap the PMI Get_clique_size API.
 *
 * @brief Wrapper for the PMI Get_clique_size API.
 *
 * @param[out] nranks_on_node The number of ranks on this node.
 *
 * @return void
 */
void ltu_pm_Clique_size(int *);

/**
 * ltu_pm_Barrier() - LTU utility function to wrap the PMI Barrier API.
 *
 * @brief Wrapper for the PMI Barrier API.
 *
 * @return void
 */
void ltu_pm_Barrier(void);

/**
 * ltu_pm_Allgather() - LTU utility function to wrap the PMI Allgather API.
 *
 * @brief Wrapper for the PMI Allgather API.
 *
 * @param[in] src The value to send to all of the other ranks.
 * @param[in] len_per_rank The length of the value to be sent.
 * @param[out] targ An array of src values from all of the ranks.
 *
 * @return void
 */
void ltu_pm_Allgather(void *src, size_t, void *dest);

/**
 * ltu_pm_Bcast() - LTU utility function to wrap the PMI Bcast API.
 *
 * @brief Wrapper for the PMI Bcast API.
 *
 * @param[in] buffer The value to be broadcast to all of the other ranks.
 * @param[in] len The length of the value to be broadcast.
 *
 * @return void
 */
void ltu_pm_Bcast(void *, size_t);

/**
 * ltu_set_pm_debug() - LTU utility function to enable and disable LTU PM debugging.
 *
 * @brief Set the LTU PM debugging level.
 *
 * @param[in] new_debug_level The LTU PM debug level that debugging information will be displayed at.
 *
 * @return void
 */

void ltu_set_pm_debug(uint32_t new_debug_level);

#endif /* LTU_UTILS_PM_H */
