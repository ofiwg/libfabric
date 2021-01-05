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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * Adapted for HPE internal use with libfabric collectives testing.
 * Joe Nemeth
 *
 * @defgroup LTU_UTILS_PM_FUNCTIONS ltu_utils_pm.c
 * @ingroup UTILITY_FUNCTIONS
 *
 * @section ltu_utils_pm_functions Overview:
 * - LTU Process Management (pm) utility functions for the Libfabric test environment.
 *
 * @section ltu_utils_pm_details Details:
 * - ltu_get_pm_version() - LTU utility function to return an integer value for the version of PM
 * - ltu_pm_Init() - LTU utility function to wrap the PMI Init API.
 * - ltu_pm_Finalize() - LTU utility function to wrap the PMI Finalize API.
 * - ltu_pm_Abort() - LTU utility function to wrap the PMI Abort API.
 * - ltu_pm_Exit() - LTU utility function to wrap the PMI Exit API.
 * - ltu_pm_Exit_with_code() - LTU utility function to wrap the PMI Exit API.
 * - ltu_pm_Rank() - LTU utility function to wrap the PMI Get_Rank API.
 * - ltu_pm_Job_size() - LTU utility function to wrap the PMI Get_Size API.
 * - ltu_pm_Clique_size() - LTU utility function to wrap the PMI Get_clique_size API.
 * - ltu_pm_Barrier() - LTU utility function to wrap the PMI Barrier API.
 * - ltu_pm_Allgather() - LTU utility function to wrap the PMI Allgather API.
 * - ltu_pm_Bcast() - LTU utility function to wrap the PMI Bcast API.
 *
 * @see ltu_utils_pm.h
 *
 * @author Anthony J. Zinger 01/30/2017.
 * @file ltu_utils_pm.c
 *
 */

#include "ltu_utils_pm.h"
#ifdef USE_PMI2 /* USE_PMI2 */
#include <slurm/pmi2.h>
#else /* NOT USE_PMI2 */
#include <slurm/pmi.h>
#endif /* END of USE_PMI2 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>
#include <dlfcn.h>

#define LTU_PM_JOB_ID_SIZE 256
#define LTU_ID_STRING_LENGTH 64
#define LTU_KEY_VALUE_LENGTH (2 * LTU_ID_STRING_LENGTH)
#define LTU_PM_STR_ID_LENGTH 48

#ifdef CRAY_PMI_COLL /* CRAY_PMI_COLL */
#ifdef USE_PMI2 /* USE_PMI2 */
#define LTU_PMI_LIBRARY "libpmi2.so.0"
#else /* NOT USE_PMI2 */
#define LTU_PMI_LIBRARY "libpmi.so.0"
#endif /* END of USE_PMI2 */

#define LTU_PMI_GET_RANK PMI_Get_rank_in_app

char ltu_pm_version[12] = "CRAY-PE-PMI";     /**< LTU PM version level. */
#else /* NOT CRAY_PMI_COLL */
#ifdef USE_PMI2 /* USE_PMI2 */
#define LTU_PMI_LIBRARY "libpmi2.so.0"
#define LTU_PMI_GET_RANK PMI2_Job_GetRank

char ltu_pm_version[12] = "SLURM-PMI-2";     /**< LTU PM version level. */
#else /* NOT USE_PMI2 */
#define LTU_PMI_LIBRARY "libpmi.so.0"
#define LTU_PMI_GET_RANK PMI_Get_rank

char ltu_pm_version[12] = "SLURM-PMI-1";     /**< LTU PM version level. */
#endif /* END of USE_PMI2 */
#endif /* END of CRAY_PMI_COLL */

#ifdef USE_PMI2 /* USE_PMI2 */
#define LTU_PMI_SUCCESS PMI2_SUCCESS
#define LTU_PMI_ABORT PMI2_Abort
#define kLTU_PMI_BARRIER PMI2_KVS_Fence
#define LTU_PMI_FINALIZE PMI2_Finalize
#define CHUNK_SIZE 511
#else /* NOT USE_PMI2 */
#define LTU_PMI_SUCCESS PMI_SUCCESS
#define LTU_PMI_ABORT PMI_Abort
#define kLTU_PMI_BARRIER PMI_Barrier
#define LTU_PMI_FINALIZE PMI_Finalize
#define CHUNK_SIZE 255

static char *kvsName;
#endif /* END of USE_PMI2 */

uint32_t ltu_pm_debug_level = 0; /**< LTU PM debug level. */
static int already_called = 0;
static char job_id[LTU_PM_JOB_ID_SIZE] = "";
static int job_size = 0;
static void *libpmi_handle = NULL;
static int *ivec_ptr = NULL;
static int myRank = -1;

/**
 * Fill out prefix for debug messages.
 *
 * @param[inout] pfx Buffer to hold prefix.
 * @param[in] len  Length of buffer for prefix.
 */
static inline void ltu_prefix(char *pfx, size_t len)
{
    snprintf(pfx, len, "# [Rank: %05i, %s] %s ",
             myRank, ltu_pm_version, "DEBUG:");
}

/**
 * ltu_pm_printf() - Format and print a message out.
 *
 * @brief Format and print a message out.
 *
 * @param[in] fmt The message format string.
 *
 * @return void
 */
__attribute__ ((format (printf, 1, 2)))
static void ltu_pm_printf(const char *fmt, ...)
{
    int ret;
    char *str;
    char str_id[LTU_PM_STR_ID_LENGTH];
    va_list args;

    ltu_prefix(str_id, LTU_PM_STR_ID_LENGTH);
    va_start(args, fmt);
    ret = vasprintf(&str, fmt, args);
    va_end(args);
    if (ret != -1) {
        fprintf(stdout, "%s%s\n", str_id, str);
        free(str);
    }
}

#ifndef CRAY_PMI_COLL /* NOT CRAY_PMI_COLL */

/** Character to use to encode and decode the message. */
#define ENCODE_CHAR    'a'

/**
 * ENCODE_LEN() - LTU utility macro to set the out of band encoding length.
 *
 * @brief The out of band encoding length.
 *
 *
 * @def ENCODE_LEN(_len) The out of band encoding length.
 *
 * @param[in] _len The encoding length to use.
 */

#define ENCODE_LEN(_len)    (2 * _len + 1)

/**
 * encode() - LTU utility function to encode a buffer.
 *
 * The underlying PMI code treats data as strings, so a NUL value terminates the
 * data. This encodes the data into half-bytes, thus doubling the size of the
 * buffer, and adds an offset to each byte to ensure that it cannot be zero.
 *
 * @brief Encode the specified buffer.
 *
 * @param[in] buf The input buffer to encode.
 * @param[in] len The length of the buffer.
 *
 * @return ebuf - The encoded buffer.
 */

static char *encode(char *buf, size_t len)
{
    int i;
    char *ebuf;

    /* need enough space for terminating NUL char */
    ebuf = calloc(ENCODE_LEN(len), 1);
    assert(ebuf);

    for (i = 0 ; i < len; i++) {
        ebuf[(2 * i)] = (buf[i] & 0xF) + ENCODE_CHAR;
        ebuf[(2 * i) + 1] = ((buf[i] >> 4) & 0xF) + ENCODE_CHAR;
    }

    ebuf[2 * len] = '\0';

    return ebuf;
}

/**
 * decode() - LTU utility function to decode a buffer.
 *
 * This presumes a well-formed encoded buffer, meaning that the NUL
 * byte value is guaranteed and represents the end of the string.
 *
 * @brief Decode the specified buffer.
 *
 * @param[in] ebuf The input buffer to decode.
 * @param[in] outlen The length of the decoded buffer.
 *
 * @return ebuf - The decoded buffer.
 */

static char *decode(char *ebuf, size_t *outlen)
{
    int i;
    char *buf;
    int len;

    /* encoded length should always be even */
    len = strlen(ebuf);
    assert(!(len & 1));

    buf = malloc(len / 2);
    assert(buf);

    for (i = 0; i < len / 2; i++) {
        buf[i] = (((ebuf[(2 * i) + 1] - ENCODE_CHAR) << 4) | (ebuf[(2 * i)] - ENCODE_CHAR));
    }

    *outlen = len / 2;

    return buf;
}

#define ROUNDUP(x, m)   ((x + (m - 1)) & ~(m - 1))

/**
 * Format and print encoded data in a readable format.
 *
 * This looks like the following:
 * ...header line...
 *  aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa | xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
 *  aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa | xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
 * ...
 *
 * The left side is encoded byte values [a-p], and the right side is hex digits.
 * Note that since the format of the data is not known (chars, bytes, ints), the
 * hex representation will be bytes in little-endian order.
 *
 * @param[in] op The opereration being encoded/decoded.
 * @param[in] encoded The encoded data buffer.
 */
static void ltu_pm_printf_encoded(char *op, char *encoded)
{
    char str_id[LTU_PM_STR_ID_LENGTH];
    unsigned char *enc;
    unsigned char *dec;
    char *str, *sp;
    size_t strsiz;
    size_t dlen;
    int i, j, k;
    int done = 0;

    enc = (unsigned char *)encoded;
    dec = (unsigned char *)decode(encoded, &dlen);

    /* Generous space. Each line of output consumes 16 bytes of decoded data,
     * and the entire line fits easily into 80 characters. The header line fits
     * easily in 256 bytes. Rounded up to the nearest 1k.
     */
    strsiz = ROUNDUP(256 + 80 * ROUNDUP(dlen, 16), 1024);
    str = malloc(strsiz);
    sp = str;

    ltu_prefix(str_id, LTU_PM_STR_ID_LENGTH);
    sp += sprintf(sp, "%s%s enc len = %ld, dec len = %ld\n",
           str_id, op, strlen(encoded), dlen);

    for (i = j = 0; j < dlen; i += 32, j += 16) {
        sp += sprintf(sp, " ");
        for (k = 0; k < 32; k++) {
            if (!done && !enc[i + k])
                done = 1;
            if (k && !(k % 8))
                sp += sprintf(sp, " ");
            sp += sprintf(sp, "%c", (!done) ? enc[i + k] : ' ');
        }
        sp += sprintf(sp, " | ");
        for (k = 0; k < 16; k++) {
            if (k && !(k % 4))
                sp += sprintf(sp, " ");
            if ((j + k) < dlen)
                sp += sprintf(sp, "%02x", dec[j + k]);
        }
        sp += sprintf(sp, "\n");
    }
    fprintf(stdout, "%s", str);
    free(str);
    free(dec);
}

/**
 * ltu_pm_send() - send a buffer.
 *
 * This sends a buffer of data associated with a key. After executing one or
 * more of these, you must issue LTU_PM_COMMIT() to ensure that all data
 * is committed. All Get operations should be preceded by LTU_PMI_BARRIER() to
 * wait for data to be committed before attempting to read it.
 *
 * PMI has a limit on the size of the send, so this sends the data in
 * chunks with different KEYS. The corresponding receive routines will
 * generate and use the appropriate keys for each chunk.
 *
 * @brief Send data.
 *
 * @param[in] kvs Operation-specific key.
 * @param[in] buffer Data buffer to send.
 * @param[in] len Length of data in the buffer (bytes).
 */
static void ltu_pm_send(char *kvs, void *buffer, size_t len)
{
    char *data;
    char key[LTU_KEY_VALUE_LENGTH];
    int __attribute__((unused)) rc = LTU_PMI_SUCCESS;
    size_t chunk, i;
    char *ptr, chr;

    /* Send data - not complete until LTU_PM_COMMIT() */
    data = encode(buffer, len);
    chunk = CHUNK_SIZE;
    ptr = data;
    for (i = 0; i < len; i += chunk) {
        /* Limit to remaining bytes */
        if (chunk > len - i)
            chunk = len - i;

        snprintf(key, sizeof(key), "%s.%ld_rank%d", kvs, i, myRank);

        /* capture this byte value and truncate string */
        chr = ptr[2*chunk];
        ptr[2*chunk] = 0;

#ifdef USE_PMI2 /* USE_PMI2 */
        rc = PMI2_KVS_Put(key, ptr);
#else /* NOT USE_PMI2 */
        rc = PMI_KVS_Put(kvsName, key, ptr);
#endif /* END of USE_PMI2 */
        assert(rc == LTU_PMI_SUCCESS);

        if (ltu_pm_debug_level > 1) {
            ltu_pm_printf("%s: KVS_Put job_id: '%s', kvs: '%s', key: '%s',"
                          " dlen: %ld, rc: %d",
                          __func__, job_id, kvs, key, chunk, rc);
        }

        /* 'untruncate' string, continue with next chunk */
        ptr[2*chunk] = chr;
        ptr += 2*chunk;
    }
    if (ltu_pm_debug_level > 1) {
        ltu_pm_printf_encoded("SEND", data);
    }
    free(data);
}

/**
 * ltu_pm_receive() - receive a buffer.
 *
 * Receive a buffer sent by the specified rank.
 *
 * This reads data sent with the associated kvs key. A node can send data
 * with a specific key. Any (or all) nodes can receive the data by using the
 * appropriate key as specified by the sender.
 *
 * This recovers data in appropriately-sized chunks.
 *
 * These should be preceded by a call to LTU_PMI_BARRIER() to ensure that data
 * is committed from the sending nodes.
 *
 * @brief Receive data.
 *
 * @param[in] kvs Operation-specific key.
 * @param[in] rank Rank of the sender.
 * @param[in] buffer Data buffer to receive.
 * @param[in] len Size of the buffer (bytes).
 */
static void ltu_pm_receive(char *kvs, int rank, void *buffer,
                           size_t len)
{
    char *data;
    char key[LTU_KEY_VALUE_LENGTH];
    int __attribute__((unused)) rc = LTU_PMI_SUCCESS;
    char *ptr;
    size_t outlen, chunk, i;
#ifdef USE_PMI2 /* USE_PMI2 */
    int retlen;
#endif /* END of USE_PMI2 */

    /* This loop reads all of the encoded data into one buffer */
    chunk = CHUNK_SIZE;
    data = calloc(ENCODE_LEN(len), 1);
    ptr = data;
    for (i = 0; i < len; i += chunk) {
        /* Limit to remaining bytes */
        if (chunk > len - i)
            chunk = len - i;

        snprintf(key, sizeof(key), "%s.%ld_rank%d", kvs, i, rank);

        /* Reads twice as many bytes as the chunk size, plus NUL */
#ifdef USE_PMI2 /* USE_PMI2 */
        rc = PMI2_KVS_Get(job_id, rank, key, ptr, 2*chunk + 1, &retlen);
        assert(!(retlen & 1));          /* count should be even */
#else /* NOT USE_PMI2 */
        rc = PMI_KVS_Get(kvsName, key, ptr, 2*chunk + 1);
#endif /* END of USE_PMI2 */
        if (ltu_pm_debug_level > 1) {
                ltu_pm_printf("%s: KVS_Get job_id: '%s', kvs: '%s', key: '%s', rc: %d",
                              __func__, job_id, kvs, key, rc);
        }
        assert(rc == LTU_PMI_SUCCESS);

        ptr += 2*chunk;
    }
    if (ltu_pm_debug_level > 1) {
        ltu_pm_printf_encoded("RECV", data);
    }

    /* Decode all of the data in one pass */
    ptr = decode(data, &outlen);
    assert(ptr != NULL);
    assert(outlen <= len);

    /* Copy decoded data into return buffer */
    memcpy(buffer, ptr, outlen);

    free(data);
    free(ptr);
}

/**
 * Perform a fencing (barrier) operation with Put commit.
 *
 * For PMI2, this is identical to LTU_PMI_BARRIER(), since the PMI2 Fence
 * operation also flushes all data Put.
 *
 * For PMI, this is distinct from LTU_PMI_BARRIER(), and should be used if you
 * have just completed one or more Put operations that other nodes will Get. The
 * Get operations should all be preceded by LTU_PMI_BARRIER(), to allow the Puts
 * to complete before they attempt to get.
 *
 */
static void LTU_PM_COMMIT(void)
{
    int __attribute__((unused)) rc = LTU_PMI_SUCCESS;
#ifdef USE_PMI2 /* USE_PMI2 */
    rc = PMI2_KVS_Fence();
    assert(rc == LTU_PMI_SUCCESS);
#else /* NOT USE_PMI2 */
    rc = PMI_KVS_Commit(kvsName);
    assert(rc == LTU_PMI_SUCCESS);
    rc = PMI_Barrier();
    assert(rc == LTU_PMI_SUCCESS);
#endif /* END of USE_PMI2 */
}

/**
 * PMI_Allgather() - LTU utility function to wrap the out of band PMI Allgather API.
 *
 * @brief Wrapper function for the out of band PMI Allgather API.
 *
 * @param[in] src The value to send to all of the other ranks.
 * @param[out] targ An array of src values from all of the ranks.
 * @param[in] len_per_rank The length of the value to be sent.
 *
 * @return PMI_SUCCESS - Operation completed successfully.
 */

int kPMI_Allgather(void *src, void *targ, size_t len_per_rank)
{
    static int cnt = 0;
    int i;
    char *ptr;
    char idstr[LTU_ID_STRING_LENGTH];
    int nranks;

    snprintf(idstr, sizeof(idstr), "allg%d", cnt++);

    /* Send data from this node */
    ltu_pm_send(idstr, src, len_per_rank);
    if (ltu_pm_debug_level > 1) {
        ltu_pm_printf("%s: send: idstr: '%s', len_per_rank: %li",
                      __func__, idstr, len_per_rank);
    }
    LTU_PM_COMMIT();
    /* All nodes Put, so all nodes have fenced. */

#ifdef USE_PMI2 /* USE_PMI2 */
    nranks = job_size;
#else /* NOT USE_PMI2 */
    PMI_Get_size(&nranks);
#endif /* END of USE_PMI2 */

    /* Collect data from all ranks into return array */
    for (i = 0; i < nranks; i++) {
        ptr = ((char *)targ) + (i * len_per_rank);
        ltu_pm_receive(idstr, i, ptr, len_per_rank);
        if (ltu_pm_debug_level > 1) {
            ltu_pm_printf("%s: receive: index: %i, ptr: %p, idstr: '%s', len_per_rank: %li",
                          __func__, i, ptr, idstr, len_per_rank);
        }
    }

    return LTU_PMI_SUCCESS;
}

/**
 * PMI_Bcast() - LTU utility function to wrap the out of band PMI Bcast API.
 *
 * @brief Wrapper function for the out of band PMI Bcast API.
 *
 * @param[in] buf The value to be broadcast to all of the other ranks.
 * @param[in] len The length of the value to be broadcast.
 *
 * @return PMI_SUCCESS - Operation completed successfully.
 */

int kPMI_Bcast(void *buf, int len)
{
    static int cnt = 0;
    char idstr[LTU_ID_STRING_LENGTH];

    snprintf(idstr, sizeof(idstr), "bcst%d", cnt++);

    if (!myRank) {
        ltu_pm_send(idstr, buf, len);
        LTU_PM_COMMIT();
    } else {
        /* receiving nodes must fence */
        kLTU_PMI_BARRIER();
        ltu_pm_receive(idstr, 0, buf, len);
    }

    kLTU_PMI_BARRIER();

    return LTU_PMI_SUCCESS;
}

#ifdef USE_PMI /* USE_PMI */
static void ltu_pm_coll_init(void)
{
    int len;
    int rank;
    int __attribute__((unused)) rc;

    rc = PMI_Get_rank(&rank);
    assert(rc == LTU_PMI_SUCCESS);

    myRank = rank;

    rc = PMI_KVS_Get_name_length_max(&len);
    assert(rc == LTU_PMI_SUCCESS);

    kvsName = calloc(len, sizeof(char));
    rc = PMI_KVS_Get_my_name(kvsName, len);
    assert(rc == LTU_PMI_SUCCESS);

    rc = PMI_Get_size(&job_size);
    assert(rc == PMI_SUCCESS);

    kLTU_PMI_BARRIER();
}
#endif /* END of USE_PMI */
#else /* CRAY_PMI_COLL */
#ifdef USE_PMI /* USE_PMI */
#define ltu_pm_coll_init()
#endif /* END of USE_PMI */
#endif /* END of CRAY_PMI_COLL */

/**
 * Helper function to ensure that allgather data is properly rank-ordered.
 *
 * This does not appear to be necessary with PMI or PMI2 implementations, since
 * they use a KVS transport that encodes the rank in the key, and then collects
 * them in rank-order. This could be necessary for other implementations.
 *
 * @param[in] in The value to send to all of the other ranks.
 * @param[out] out An array of src values from all of the ranks.
 * @param[in] len The length of the value to be sent by one rank.
 */
static void allgather(void *in, void *out, int len)
{
    int i, __attribute__((unused)) rc;
    char *tmp_buf, *out_ptr;

    if (!already_called) {
        ivec_ptr = (int *)calloc(job_size, sizeof(int));
        assert(ivec_ptr != NULL);

        rc = kPMI_Allgather(&myRank, ivec_ptr, sizeof(int));
        assert(rc == LTU_PMI_SUCCESS);

        already_called = 1;
    }

    if (ltu_pm_debug_level > 1) {
        ltu_pm_printf("%s: job_size: %i, len: %i",
                      __func__, job_size, len);
    }

    tmp_buf = calloc(job_size, len);
    assert(tmp_buf);

    rc = kPMI_Allgather(in, tmp_buf, len);
    assert(rc == LTU_PMI_SUCCESS);

    out_ptr = out;

    for (i = 0; i < job_size; i++) {
        memcpy(&out_ptr[len * ivec_ptr[i]], &tmp_buf[i * len], len);

        if (ltu_pm_debug_level > 1) {
            ltu_pm_printf("%s: copy out: ivec_ptr: %i, len: %i",
                          __func__, ivec_ptr[i], len);
        }
    }

    free(tmp_buf);
}

/* exported functions */

/**
 * ltu_set_pm_debug() - LTU utility function to enable and disable LTU PM debugging.
 *
 * @brief Set the LTU PM debugging level.
 *
 * @param[in] new_debug_level The LTU PM debug level that debugging information will be displayed at.
 *
 * @return void
 */

void ltu_set_pm_debug(uint32_t new_debug_level)
{
    ltu_pm_debug_level = new_debug_level;
}

/**
 * ltu_get_pm_version() - LTU utility function to return an integer value for the version of PM
 *
 * @brief Return an integer value for the version of PM.
 *
 * @return string - containing the version of PM.
 */
char * ltu_get_pm_version(void)
{
    return (ltu_pm_version);
}

/**
 * ltu_pm_Init() - LTU utility function to wrap the PMI2 Init API.
 *
 * @brief Wrapper for the PMI2 Init API.
 *
 * @param[in] argc The application's argc parameter.
 * @param[in] argv The application's argv parameter.
 *
 * @return void
 */

void ltu_pm_Init(int *argc, char ***argv)
{
    int __attribute__((unused)) rc;
#ifdef USE_PMI2 /* USE_PMI2 */
    int appnum;
#endif /* END of USE_PMI2 */
    int spawned;

    libpmi_handle = dlopen(LTU_PMI_LIBRARY, RTLD_LAZY | RTLD_GLOBAL);
    if (libpmi_handle == NULL) {
        perror("Unable to open libpmi.so check your LD_LIBRARY_PATH");
        abort();
    }

#ifdef USE_PMI2 /* USE_PMI2 */
    rc = PMI2_Init(&spawned, &job_size, &myRank, &appnum);
    assert(rc == LTU_PMI_SUCCESS);
    memset((void *) &job_id, 0, LTU_PM_JOB_ID_SIZE);
    rc = PMI2_Job_GetId(job_id, LTU_PM_JOB_ID_SIZE);
    if (rc != LTU_PMI_SUCCESS)
        fprintf(stderr, "**** must be run under a workload manager ****\n");
    assert(rc == LTU_PMI_SUCCESS);

    if (ltu_pm_debug_level > 0) {
        ltu_pm_printf("%s: job_size: %i, rank %i, job_id '%s', appnum: %i",
                      __func__, job_size, myRank, job_id, appnum);
    }
#else /* NOT USE_PMI2 */
    rc = PMI_Init(&spawned);
    assert(rc == LTU_PMI_SUCCESS);

    ltu_pm_coll_init();

    if (ltu_pm_debug_level > 0) {
        ltu_pm_printf("%s: job_size: %i, rank %i, job_id '%s'",
                      __func__, job_size, myRank, job_id);
    }
#endif /* END of USE_PMI2 */
}

/**
 * ltu_pm_Finalize() - LTU utility function to wrap the PMI Finalize API.
 *
 * @brief Wrapper for the PMI Finalize API.
 *
 * @return void
 */

void ltu_pm_Finalize(void)
{
    if (ivec_ptr != NULL ) {
        free(ivec_ptr);
        ivec_ptr = NULL;
        already_called = 0;
        job_size = 0;
        myRank = -1;
        memset((void *) &job_id, 0, LTU_PM_JOB_ID_SIZE);
    }

    LTU_PMI_FINALIZE();

    if (libpmi_handle != NULL ) {
        dlclose(libpmi_handle);
        libpmi_handle = NULL;
    }
}

/**
 * ltu_pm_Abort() - LTU utility function to wrap the PMI Abort API.
 *
 * @brief Wrapper for the PMI Abort API.
 *
 * @return void
 */

void ltu_pm_Abort(void)
{
    LTU_PMI_ABORT(-1, "pmi abort called");
}

/**
 * ltu_pm_Exit() - LTU utility function to wrap the PMI Exit API.
 *
 * @brief Wrapper for the PMI Exit API.
 *
 * @return void
 */

void ltu_pm_Exit(void)
{
    LTU_PMI_ABORT(0, "Terminating application successfully");
}

/**
 * ltu_pm_Exit_with_code() - LTU utility function to wrap the PMI Exit API.
 *
 * @brief Wrapper for the PMI Exit API.
 *
 * @param[in] code The return code that this job will exit with.
 *
 * @return void
 */

void ltu_pm_Exit_with_code(int code)
{
    LTU_PMI_ABORT(code, "Terminating application with return code");
}

/**
 * ltu_pm_Rank() - LTU utility function to wrap the PMI Get Rank API.
 *
 * @brief Wrapper for the PMI2 Get Rank API.
 *
 * @param[out] rank The PMI rank of this process.
 *
 * @return void
 */

void ltu_pm_Rank(int *rank)
{
    int __attribute__((unused)) rc;

    rc = LTU_PMI_GET_RANK(rank);
    assert(rc == LTU_PMI_SUCCESS);

    if (ltu_pm_debug_level > 0) {
        ltu_pm_printf("%s: Rank: %i", __func__, *rank);
    }
}

/**
 * ltu_pm_Job_size() - LTU utility function to wrap the PMI Get Job Size API.
 *
 * @brief Wrapper for the PMI Get Job Size API.
 *
 * @param[out] nranks The number of ranks of this job.
 *
 * @return void
 */

void ltu_pm_Job_size(int *nranks)
{
#ifdef USE_PMI2 /* USE_PMI2 */
    *nranks = job_size;
#else /* NOT USE_PMI2 */
    int __attribute__((unused)) rc;

    rc = PMI_Get_size(nranks);
    assert(rc == PMI_SUCCESS);
#endif /* END of USE_PMI2 */

    if (ltu_pm_debug_level > 0) {
        ltu_pm_printf("%s: Job_size: number of ranks: %i", __func__, *nranks);
    }
}

/**
 * ltu_pm_Clique_size() - LTU utility function to wrap the PMI Get Ranks on Node Size API.
 *
 * @brief Wrapper for the PMI Get Ranks on Node Size API.
 *
 * @param[out] nranks_on_node The number of ranks on this node.
 *
 * @return void
 */

void ltu_pm_Clique_size(int *nranks_on_node)
{
#ifdef USE_PMI2 /* USE_PMI2 */
#ifndef CRAY_PMI_COLL /* CRAY_PMI_COLL */
    int array_len = 20;
    int array_values[20] = {0};
#endif /* END of CRAY_PMI_COLL */
    const char *attr_name = "localRanksCount";
    int found = 0;
    int outlen = 0;
#endif /* END of USE_PMI2 */
    int __attribute__((unused)) rc;

#ifdef USE_PMI2 /* USE_PMI2 */
#ifdef CRAY_PMI_COLL /* CRAY_PMI_COLL */
    rc = PMI_Get_clique_size(nranks_on_node);
    assert(rc == LTU_PMI_SUCCESS);
#else /* NOT CRAY_PMI_COLL */
    rc = PMI2_Info_GetNodeAttrIntArray(attr_name, array_values, array_len, &outlen, &found);
    assert(rc == LTU_PMI_SUCCESS);
    if (found > 0) {
        *nranks_on_node = array_values[0];
    } else {
        *nranks_on_node = 1;
    }
#endif /* END of CRAY_PMI_COLL */
#else /* NOT USE_PMI2 */
    rc = PMI_Get_clique_size(nranks_on_node);
    assert(rc == LTU_PMI_SUCCESS);
#endif /* END of USE_PMI2 */

    if (ltu_pm_debug_level > 0) {
#ifdef USE_PMI2 /* USE_PMI2 */
        ltu_pm_printf("%s: found: %i, attr_name: '%s', outlen: %i, number of ranks on node: %i",
                      __func__, found, attr_name, outlen, *nranks_on_node);
#else /* NOT USE_PMI2 */
        ltu_pm_printf("%s: number of ranks on node: %i",
                      __func__, *nranks_on_node);
#endif /* END of USE_PMI2 */
    }
}

/**
 * ltu_pm_Barrier() - LTU utility function to wrap the PMI Barrier API.
 *
 * @brief Wrapper for the PMI Barrier API.
 *
 * @return void
 */

void ltu_pm_Barrier(void)
{
    int __attribute__((unused)) rc;

    rc = kLTU_PMI_BARRIER();
    assert(rc == LTU_PMI_SUCCESS);
}

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

void ltu_pm_Allgather(void *src, size_t len_per_rank, void *targ)
{
    allgather(src, targ, len_per_rank);
}

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

void ltu_pm_Bcast(void *buffer, size_t len)
{
    int __attribute__((unused)) rc;

    rc = kPMI_Bcast(buffer, len);
    assert(rc == LTU_PMI_SUCCESS);
}
