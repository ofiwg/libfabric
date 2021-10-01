/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2014 Intel Corporation, Inc. All rights reserved.
 * Copyright (c) 2016 Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2017 DataDirect Networks, Inc. All rights reserved.
 * Copyright (c) 2016-2021 Cray Inc. All rights reserved.
 */

/**
 * Adapted for HPE internal use with libfabric collectives testing.
 *
 * @defgroup PMI_UTILS_FUNCTIONS pmi_utils.c
 * @ingroup UTILITY_FUNCTIONS
 *
 * @section Overview:
 * PMI Process Management (pm) utility functions for the Libfabric test
 * environment.
 *
 * PMI is a fairly primitive intercommunicator that works a bit like an indexed
 * database. Processes can "put" to the database using a key, and other
 * processes can retrieve the data using the same key.
 *
 * You cannot overwrite data associated with a key, and you cannot delete old
 * keys, so every new write requires a new key. There is a limit on the number
 * of keys supported, governed by environment variable PMI_MAX_KVS_ENTRIES.
 * If not specified, it is scaled by the WLM/PMI primitives to be able to do
 * a little more than one allgather.
 *
 * This layer implements the MPI-like features of Barrier, Broadcast, and
 * Allgather, using the underlying PMI layer. It isn't particularly fast, but
 * it can be used to set up a multi-node test environment.
 *
 * This layer is designed to work with different PMI implementations, by using
 * different sections of code to implement "shims" that convert from that PMI
 * to a generic shim layer of code. Only the CRAY PMI2 implementation is
 * currently supported.
 *
 * @section pmi dependency shims:
 * Every PMI version has its own way of doing things. This creates the common
 * code used by all version, for use in the exported pmi utility functions.
 * - pmi_shim_Init     - initialize PMI system
 * - pmi_shim_Finalize - terminate PMI system
 * - pmi_shim_Commit   - flush and synchronize
 * - pmi_shim_Fence  - synchronize
 * - pmi_shim_Put      - put data to common area
 * - pmi_shim_Get      - get data from common area
 * - pmi_shim_Abort    - kill all processes and exit
 *
 * @section implementation helper functions
 * Code for adapting PMI shims to mpi-like functions.
 *
 * @section pmi utility library implementation:
 * Exported functions. See pmi_utils.h.
 *
 * @see pmi_utils.h
 *
 * @author Joseph Nemeth 10/14/2021.
 * @author based on work by Anthony J. Zinger 01/30/2017.
 *
 * @file pmi_utils.c
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>

#define PMI_JOB_ID_SIZE 256
#define PMI_ID_STRING_LENGTH 64
#define PMI_KEY_VALUE_LENGTH (2 * PMI_ID_STRING_LENGTH)
#define PMI_PM_DBG_ID_LENGTH 48

/* Forward-references */
static void pmi_shim_Init(int *numranks, int *rank, int *appnum);
static void pmi_shim_Finalize(void);
static void pmi_shim_Commit(void);
static void pmi_shim_Fence(void);
static void pmi_shim_Put(char *key, char *ptr);
static void pmi_shim_Get(const char *key, char *buf, int size);
static void pmi_shim_Abort(int error, const char *msg);
static void pmi_errmsg(const char *fmt, ...);
static void pmi_dbgmsg(int level, const char *fmt, ...);

static char *pmi_version;
static bool pmi_initialized;
static char pmi_jobid[256];
static int pmi_numranks;
static int pmi_rank;
static int pmi_appnum;
static int pmi_kvs_id;
static int pmi_debug_level;

/**
 * PMI shim layer. Each PMI library has idiosyncracies.
 */

/***************************************************/
#if defined(USE_CRAY_PMI) && defined (USE_PMI2)

#define CHUNK_SIZE 511

/**
 * This requires building with:
 *   CPPFLAGS += -DUSE_PMI2 -DUSE_CRAY_PMI
 *   CPPFLAGS += -I /opt/cray/pe/pmi/default/include
 *   LDADD += -lpmi2
 *   dynamic linking with /opt/cray/pe/pmi/default/lib
 */
#include <pmi2.h>

static char *pmi_version = "CRAY-PE-PMI2";

static void pmi_shim_Init(int *numranks, int *rank, int *appnum)
{
    int rc, spawned;

    if (pmi_initialized) {
        pmi_dbgmsg(1, "%s: already initialized\n", __func__);
        goto done;
    }

    memset((void *) pmi_jobid, 0, sizeof(pmi_jobid));

    /* Initialize this PMI */
    rc = PMI2_Init(&spawned, &pmi_numranks, &pmi_rank, &pmi_appnum);
    if (rc != PMI2_SUCCESS)
        pmi_shim_Abort(rc, "PMI2_Init failed\n");

    /* Capture job ID -- must be run under WLM */
    rc = PMI2_Job_GetId(pmi_jobid, sizeof(pmi_jobid));
    if (rc != PMI2_SUCCESS)
        pmi_shim_Abort(rc, "PMI2_Job_GetId failed\n");

    /* Prevent global re-initialization */
    pmi_initialized = true;

done:
    if (numranks)
        *numranks = pmi_numranks;
    if (rank)
        *rank = pmi_rank;
    if (appnum)
        *appnum = pmi_appnum;
    pmi_dbgmsg(1, "%s: numranks=%d rank=%d appnum=%d job='%s'\n",
                  __func__, pmi_numranks, pmi_rank, pmi_appnum, pmi_jobid);
}

static void pmi_shim_Finalize(void)
{
    int rc;

    if (! pmi_initialized)
        return;

    rc = PMI2_Finalize();
    if (rc != PMI2_SUCCESS)
        pmi_shim_Abort(rc, "pmi_shim_Finalize failed\n");

    pmi_initialized = false;
}

static void pmi_shim_Fence(void)
{
    int rc;

    rc = PMI2_KVS_Fence();
    if (rc != PMI2_SUCCESS)
        pmi_shim_Abort(rc, "pmi_shim_Fence failed\n");
}

static inline void pmi_shim_Commit(void)
{
    int rc;

    rc = PMI2_KVS_Fence();
    if (rc != PMI2_SUCCESS)
        pmi_shim_Abort(rc, "pmi_shim_Commit failed\n");
}

static inline void pmi_shim_Put(char *key, char *ptr)
{
    int rc;

    rc = PMI2_KVS_Put(key, ptr);
    if (rc != PMI2_SUCCESS)
        pmi_shim_Abort(rc, "pmi_shim_Put failed\n");
}

static inline void pmi_shim_Get(const char *key, char *buf, int size)
{
    int retlen, rc;
    rc = PMI2_KVS_Get(NULL, PMI2_ID_NULL, key, buf, size, &retlen);
    if (rc != PMI2_SUCCESS)
        pmi_shim_Abort(rc, "pmi_shim_Get failed\n");
}

static void pmi_shim_Abort(int error, const char *msg)
{
    PMI2_Abort(error, msg);
}

/***************************************************/
#elif defined(USE_CRAY_PMI)
#   error "USE_CRAY_PMI and not USE_PMI2 unsupported"
/***************************************************/
#elif defined(USE_PMI2)
#   error "not USE_CRAY_PMI and USE_PMI2 unsupported"
/***************************************************/
#else
#   error "not USE_CRAY_PMI and not USE_PMI2 unsupported"
/***************************************************/
#endif

/**
 * pmi_fprintf() variants - Format and print a message.
 */
static inline int pmi_dbg_prefix(char *pfx, size_t len, char *typ)
{
    return snprintf(pfx, len, "# [Rank: %05i, %s] %s:",
                    pmi_rank, pmi_version, typ);
}

static void pmi_fprintf(FILE *fd, char *typ, const char *fmt, va_list args)
{
    int off;
    char str[1024];

    off = 0;
    off += pmi_dbg_prefix(&str[off], sizeof(str) - off, typ);
    off += vsnprintf(&str[off], sizeof(str) - off, fmt, args);
    fprintf(fd, "%s", str);
}

__attribute__ ((format (printf, 1, 2), unused))
static void pmi_errmsg(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    pmi_fprintf(stderr, "ERROR", fmt, args);
    va_end(args);
}

__attribute__ ((format (printf, 2, 3), unused))
static void pmi_dbgmsg(int level, const char *fmt, ...)
{
    va_list args;
    if (pmi_debug_level >= level) {
        va_start(args, fmt);
        pmi_fprintf(stdout, "DEBUG", fmt, args);
        va_end(args);
    }
}

/** Character to use to encode and decode the message. */
#define ENCODE_CHAR    'a'

/**
 * ENCODE_LEN() - PMI utility macro to set the out of band encoding length.
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
 * encode() - PMI utility function to encode a buffer.
 *
 * The underlying PMI code treats data as strings, so a NUL value terminates the
 * data. This encodes the data into half-bytes, thus doubling the size of the
 * buffer, and adds an offset to each byte to ensure that it cannot be zero.
 *
 * Returned value must be freed.
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
 * decode() - PMI utility function to decode a buffer.
 *
 * This presumes a well-formed encoded buffer, meaning that the NUL
 * byte value is guaranteed and represents the end of the string.
 *
 * Returned buffer must be freed.
 *
 * @brief Decode the specified buffer.
 *
 * @param[in] ebuf The input buffer to decode.
 * @param[in] outlen The length of the decoded buffer.
 *
 * @return buf - The decoded buffer.
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
        buf[i] = (((ebuf[(2 * i) + 1] - ENCODE_CHAR) << 4) |
                 (ebuf[(2 * i)] - ENCODE_CHAR));
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
static void pmi_printf_encoded(char *op, char *encoded)
{
    char str_pfx[PMI_PM_DBG_ID_LENGTH];
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

    pmi_dbg_prefix(str_pfx, PMI_PM_DBG_ID_LENGTH, "DEBUG");
    sp += sprintf(sp, "%s%s enc len = %ld, dec len = %ld\n",
           str_pfx, op, strlen(encoded), dlen);

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
 * pmi_send() - send a buffer.
 *
 * This sends a buffer of data associated with a key. After executing one or
 * more of these, you must issue pmi_shim_Commit() to ensure that all data
 * is committed. All Get operations should be preceded by pmi_shim_Fence() to
 * wait for data to be committed before attempting to read it.
 *
 * PMI has a limit on the size of the send, so this sends the data in
 * chunks with different KEYS. The corresponding receive routines will
 * generate and use the appropriate keys for each chunk.
 *
 * This means that sending large messages will use up more KVS buffers.
 *
 * @brief Send data.
 *
 * @param[in] kvs_id KVS identifier.
 * @param[in] buffer Data buffer to send.
 * @param[in] len Length of data in the buffer (bytes).
 */
static void pmi_send(int kvs_id, int src_rank, void *buffer, size_t len)
{
    char *data;
    char key[PMI_KEY_VALUE_LENGTH];
    size_t chunk, i;
    char *ptr, chr;

    /* Send data - not complete until pmi_shim_Commit() */
    data = encode(buffer, len);
    chunk = CHUNK_SIZE;
    ptr = data;
    for (i = 0; i < len; i += chunk) {
        /* Limit to remaining bytes */
        if (chunk > len - i)
            chunk = len - i;

        snprintf(key, sizeof(key), "%d.%ld.%d", kvs_id, i, src_rank);

        /* capture this byte value and truncate string */
        chr = ptr[2*chunk];
        ptr[2*chunk] = 0;

        pmi_shim_Put(key, ptr);

        pmi_dbgmsg(3, "%s: KVS_Put job_id: '%s', kvs_id: %d,"
                        " key: '%s', dlen: %ld\n",
                        __func__, pmi_jobid, kvs_id, key, chunk);

        /* 'untruncate' string, continue with next chunk */
        ptr[2*chunk] = chr;
        ptr += 2*chunk;
    }
    if (pmi_debug_level > 1) {
        pmi_printf_encoded("SEND", data);
    }
    free(data);
}

/**
 * pmi_receive() - receive a buffer.
 *
 * Receive a buffer sent by the specified rank.
 *
 * This reads data sent with the associated kvs key. A node can send data
 * with a specific key. Any (or all) nodes can receive the data by using the
 * appropriate key as specified by the sender.
 *
 * This recovers data in appropriately-sized chunks.
 *
 * These should be preceded by a call to pmi_shim_Fence() to ensure that data
 * is committed from the sending nodes.
 *
 * @brief Receive data.
 *
 * @param[in] kvs_id KVS identifier.
 * @param[in] rank Rank of the sender.
 * @param[in] buffer Data buffer to receive.
 * @param[in] len Size of the buffer (bytes).
 */
static void pmi_receive(int kvs_id, int sendrank, void *buffer, size_t len)
{
    char *data;
    char key[PMI_KEY_VALUE_LENGTH];
    char *ptr;
    size_t outlen, chunk, i;

    /* This loop reads all of the encoded data into one buffer */
    chunk = CHUNK_SIZE;
    data = calloc(ENCODE_LEN(len), 1);
    ptr = data;
    for (i = 0; i < len; i += chunk) {
        /* Limit to remaining bytes */
        if (chunk > len - i)
            chunk = len - i;

        snprintf(key, sizeof(key), "%d.%ld.%d", kvs_id, i, sendrank);

        /* Reads twice as many bytes as the chunk size, plus NUL */
        pmi_shim_Get(key, ptr, 2*chunk + 1);
        pmi_dbgmsg(3, "%s: KVS_Get job_id: '%s', kvs_id: %d,"
                      " key: '%s'\n",
                      __func__, pmi_jobid, kvs_id, key);

        ptr += 2*chunk;
    }
    if (pmi_debug_level > 1) {
        pmi_printf_encoded("RECV", data);
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
 * Exported functions, see header file.
 */
void pmi_set_debug(int new_debug_level)
{
    pmi_debug_level = new_debug_level;
}

__attribute__ ((format (printf, 2, 3)))
void pmi_Abort(int error, const char *fmt, ...)
{
    va_list args;
    char str[1024];

    va_start(args, fmt);
    vsnprintf(str, sizeof(str)-1, fmt, args);
    va_end(args);
    pmi_shim_Abort(error, str);
}

void pmi_Init(int *numranks, int *rank, int *appnum)
{
    pmi_dbgmsg(1, "%s\n", __func__);
    pmi_shim_Init(numranks, rank, appnum);
}

void pmi_Finalize(void)
{
    pmi_dbgmsg(1, "%s\n", __func__);
    pmi_shim_Finalize();
}

void pmi_GetRank(int *rank)
{
    *rank = pmi_rank;
    pmi_dbgmsg(1, "%s: Rank: %d\n", __func__, *rank);
}

void pmi_GetNumRanks(int *numranks)
{
    *numranks = pmi_numranks;
    pmi_dbgmsg(1, "%s: NumRanks: %d\n", __func__, *numranks);
}

int pmi_GetKVSCount(size_t bufsize)
{
    return ENCODE_LEN(bufsize) / CHUNK_SIZE;
}

void pmi_Barrier(void)
{
    pmi_dbgmsg(1, "%s\n", __func__);
    pmi_shim_Fence();
}

void pmi_Bcast(int src_rank, void *buffer, size_t len)
{
    int kvs_id = pmi_kvs_id++;

    pmi_dbgmsg(1, "%s: src_rank=%d, len=%ld\n", __func__, src_rank, len);
    if (pmi_rank == src_rank) {
        /* sendrank is arbitrary, but must match receive */
        pmi_send(kvs_id, 0, buffer, len);
        pmi_dbgmsg(2, "%s: send kvs_id=%d, len=%ld\n",
                      __func__, kvs_id, len);
        /* flush and commit */
        pmi_shim_Commit();
    } else {
        /* receiving nodes must fence before receive */
        pmi_shim_Fence();
        pmi_receive(kvs_id, 0, buffer, len);
        pmi_dbgmsg(2, "%s: recv kvs_id=%d, len=%ld\n",
                      __func__, kvs_id, len);
    }
    pmi_shim_Fence();
}

void pmi_Allgather(void *srcbuf, size_t len_per_rank, void *tgtbuf)
{
    int kvs_id = pmi_kvs_id++;
    int i;
    uint8_t *ptr;

    pmi_dbgmsg(1, "%s: len=%ld\n", __func__, len_per_rank);

    /* Send data from this node */
    pmi_send(kvs_id, pmi_rank, srcbuf, len_per_rank);
    pmi_dbgmsg(2, "%s: send kvs_id=%d, rank=%d, len=%ld\n",
                  __func__, kvs_id, pmi_rank, len_per_rank);

    pmi_shim_Commit();
    /* All nodes Put, so all nodes have fenced. */

    /* Collect data from all ranks into return array */
    ptr = (uint8_t *)tgtbuf;
    for (i = 0; i < pmi_numranks; i++) {
        pmi_receive(kvs_id, i, ptr, len_per_rank);
        pmi_dbgmsg(2, "%s: recv index=%d, ptr=%p, "
                      "kvs_id=%d, len=%ld\n",
                      __func__, i, ptr, kvs_id, len_per_rank);
        ptr += len_per_rank;
    }
    pmi_shim_Fence();
}
