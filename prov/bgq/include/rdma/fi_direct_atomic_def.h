#ifndef _FI_BGQ_DIRECT_ATOMIC_DEF_H_
#define _FI_BGQ_DIRECT_ATOMIC_DEF_H_

#ifdef FABRIC_DIRECT
#define FABRIC_DIRECT_ATOMIC_DEF 1

enum fi_datatype {
	FI_INT8,			/*  0 */
	FI_UINT8,			/*  1 */
	FI_INT16,			/*  2 */
	FI_UINT16,			/*  3 */
	FI_INT32,			/*  4 */
	FI_UINT32,			/*  5 */
	FI_INT64,			/*  7 */
	FI_UINT64,			/*  8 */
	FI_FLOAT,			/*  6 */
	FI_DOUBLE,			/*  9 */
	FI_FLOAT_COMPLEX,		/* 10 */
	FI_DOUBLE_COMPLEX,		/* 11 */
	FI_LONG_DOUBLE,			/* 12 */
	FI_LONG_DOUBLE_COMPLEX,		/* 13 */
	FI_DATATYPE_LAST		/* 14 */
};
enum fi_op {
	FI_MIN,
	FI_MAX,
	FI_SUM,
	FI_PROD,
	FI_LOR,
	FI_LAND,
	FI_BOR,
	FI_BAND,
	FI_LXOR,
	FI_BXOR,
	FI_ATOMIC_READ,
	FI_ATOMIC_WRITE,
	FI_CSWAP,
	FI_CSWAP_NE,
	FI_CSWAP_LE,
	FI_CSWAP_LT,
	FI_CSWAP_GE,
	FI_CSWAP_GT,
	FI_MSWAP,
	FI_ATOMIC_OP_LAST
};
#endif


#endif /* _FI_BGQ_DIRECT_ATOMIC_DEF_H_ */
