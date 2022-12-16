#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_atomic.h>

#include "shared.h"
#include "hmem.h"
#include "rdma/fi_domain.h"

struct atomic_dv_summary {
	enum fi_datatype datatype;
	enum fi_op op;
	size_t trials;
	size_t validation_failures;
	size_t validations_performed;
	size_t first_failure;
	size_t last_failure;
	struct atomic_dv_summary *next;
};

struct atomic_dv_summary* dv_summary_root = NULL;

/*
 @brief Prints a summary of test failures.
 @return 0 if all validations passed, <0 if any failures recorded.
*/
int atomic_data_validation_print_summary() {

	int retval = 0;
	char type_str[32] = {0};
	char op_str[32] = {0};
	char test_name[64] = {};

	struct atomic_dv_summary *node = dv_summary_root;
	struct atomic_dv_summary *next = NULL;

	if (!node) return 0;

	while(node) {
		snprintf(type_str, sizeof(type_str)-1, "%s", fi_tostr(&node->datatype, FI_TYPE_ATOMIC_TYPE));
		snprintf(op_str, sizeof(op_str)-1, "%s", fi_tostr(&node->op, FI_TYPE_ATOMIC_OP));
		snprintf(test_name, sizeof(test_name), "%s on %s", op_str, type_str);
		if (node->validation_failures==0 && node->validations_performed==node->trials) {
			// all these tests passed
			//printf("PASSED: %s passed %zu trials.\n",test_name, node->trials);
		}
		else if (node->validation_failures) {
			printf("FAILED: %s had %zu of %zu tests fail data validation.\n",
				test_name, node->validation_failures, node->trials);
			printf("\t\tFirst failure at trial %zu, last failure at trial %zu.\n",
				node->first_failure, node->last_failure);
			retval = -1;
		}
		else if (node->validations_performed < node->trials) {
			printf("SKIPPED: Data validation not available for %s\n", test_name);
			retval = -1;
		}

		// clean up as we go
		next = node->next;
		free(node);
		node = next;
	}

	return retval;
}

static void atomic_dv_record(enum fi_datatype dtype, enum fi_op op, bool failed, bool checked) {

	struct atomic_dv_summary *node = dv_summary_root;

	if (!node || node->op != op || node->datatype != dtype) {
		// allocate and add a new node
		node = calloc(1, sizeof(struct atomic_dv_summary));
		node->next = dv_summary_root;
		dv_summary_root = node;
		node->op = op;
		node->datatype = dtype;
	}

	// record trial.
	node->trials++;
	if (failed) {
		if (node->validation_failures==0) node->first_failure = node->trials;
		node->last_failure = node->trials;
		node->validation_failures++;
	}
	if (checked) node->validations_performed++;
}


// debugging macro help: gcc -Iinclude -I/fsx/lrbison/libfabric/install/include -E functional/atomic_verify.c | sed 's/case/\ncase/g' | less

#define ATOM_FOR_FI_MIN(a,ao,b)  (ao) = (((a) < (b)) ? a : b)
#define ATOM_FOR_FI_MAX(a,ao,b)  (ao) = (((a) > (b)) ? a : b)
#define ATOM_FOR_FI_SUM(a,ao,b)  (ao) = ((a) + (b))
#define ATOM_FOR_FI_PROD(a,ao,b) (ao) = ((a) * (b))
#define ATOM_FOR_FI_LOR(a,ao,b)  (ao) = ((a) || (b))
#define ATOM_FOR_FI_LAND(a,ao,b) (ao) = ((a) && (b))
#define ATOM_FOR_FI_BOR(a,ao,b)  (ao) = ((a) | (b))
#define ATOM_FOR_FI_BAND(a,ao,b) (ao) = ((a) & (b))
#define ATOM_FOR_FI_LXOR(a,ao,b) (ao) = (((a) && !(b)) || (!(a) && (b)))
#define ATOM_FOR_FI_BXOR(a,ao,b) (ao) = ((a) ^ (b))
#define ATOM_FOR_FI_ATOMIC_READ(a,ao,b) {}
#define ATOM_FOR_FI_ATOMIC_WRITE(a,ao,b) (ao) = (b)

#define ATOM_FOR_FI_CSWAP(a,ao,b,c)    if ((c) == (a)) {(ao) = (b);}
#define ATOM_FOR_FI_CSWAP_NE(a,ao,b,c) if ((c) != (a)) {(ao) = (b);}
#define ATOM_FOR_FI_CSWAP_LE(a,ao,b,c) if ((c) <= (a)) {(ao) = (b);}
#define ATOM_FOR_FI_CSWAP_LT(a,ao,b,c) if ((c) <  (a)) {(ao) = (b);}
#define ATOM_FOR_FI_CSWAP_GE(a,ao,b,c) if ((c) >= (a)) {(ao) = (b);}
#define ATOM_FOR_FI_CSWAP_GT(a,ao,b,c) if ((c) >  (a)) {(ao) = (b);}
#define ATOM_FOR_FI_MSWAP(a,ao,b,c)    (ao) = ((b) & (c)) | ((a) & ~(c));

#define ATOM_FOR_CPLX_FI_MIN(a,ao,b,absfun)  (ao) = ((absfun(a) < absfun(b)) ? (a) : (b))
#define ATOM_FOR_CPLX_FI_MAX(a,ao,b,absfun)  (ao) = (absfun(a) > absfun(b) ? (a) : (b))
#define ATOM_FOR_CPLX_FI_CSWAP_LE(a,ao,b,c,absfun) if (absfun(c) <= absfun(a)) {(ao) = (b);}
#define ATOM_FOR_CPLX_FI_CSWAP_LT(a,ao,b,c,absfun) if (absfun(c) <  absfun(a)) {(ao) = (b);}
#define ATOM_FOR_CPLX_FI_CSWAP_GE(a,ao,b,c,absfun) if (absfun(c) >= absfun(a)) {(ao) = (b);}
#define ATOM_FOR_CPLX_FI_CSWAP_GT(a,ao,b,c,absfun) if (absfun(c) >  absfun(a)) {(ao) = (b);}


#define ATOM_CTYPE_FOR_FI_INT32 int32_t
#define ATOM_CTYPE_FOR_FI_INT16 int16_t
#define ATOM_CTYPE_FOR_FI_INT8 int8_t
#define ATOM_CTYPE_FOR_FI_INT64 int64_t
#define ATOM_CTYPE_FOR_FI_UINT32 uint32_t
#define ATOM_CTYPE_FOR_FI_UINT16 uint16_t
#define ATOM_CTYPE_FOR_FI_UINT8 uint8_t
#define ATOM_CTYPE_FOR_FI_UINT64 uint64_t
#define ATOM_CTYPE_FOR_FI_FLOAT float
#define ATOM_CTYPE_FOR_FI_DOUBLE double
#define ATOM_CTYPE_FOR_FI_LONG_DOUBLE long double

#define ATOM_CTYPE_FOR_FI_FLOAT_COMPLEX float complex
#define ATOM_CTYPE_FOR_FI_DOUBLE_COMPLEX double complex
#define ATOM_CTYPE_FOR_FI_LONG_DOUBLE_COMPLEX long double complex

// this macro is for expansion inside the perform_atomic_op function
// and uses variables local to that function.
#define atomic_case_cplx(ftype, fop, absfun)					\
case ftype*FI_ATOMIC_OP_LAST + fop:				                \
	{   if(result) *(ATOM_CTYPE_FOR_##ftype*)result = *(ATOM_CTYPE_FOR_##ftype*)addr_in;	\
		ATOM_FOR_CPLX_##fop( *(ATOM_CTYPE_FOR_##ftype*)addr_in,		\
						 *(ATOM_CTYPE_FOR_##ftype*)addr_out,	\
						 *(ATOM_CTYPE_FOR_##ftype*)buf,	\
						 absfun );			\
		break;								\
	}
#define atomic_case(ftype, fop)							\
case ftype*FI_ATOMIC_OP_LAST + fop:						\
	{   if(result) *(ATOM_CTYPE_FOR_##ftype*)result = *(ATOM_CTYPE_FOR_##ftype*)addr_in;	\
		ATOM_FOR_##fop(  *(ATOM_CTYPE_FOR_##ftype*)addr_in,		\
						 *(ATOM_CTYPE_FOR_##ftype*)addr_out,	\
						 *(ATOM_CTYPE_FOR_##ftype*)buf );	\
		break;								\
	}

// this macro is for expansion inside the perform_atomic_op function
// and uses variables local to that function.
#define atomic_case_compare(ftype, fop)						\
case ftype*FI_ATOMIC_OP_LAST + fop:						\
	{   if(result) {*(ATOM_CTYPE_FOR_##ftype*)result = *(ATOM_CTYPE_FOR_##ftype*)addr_in; }	\
		ATOM_FOR_##fop(  *(ATOM_CTYPE_FOR_##ftype*)addr_in,		\
						 *(ATOM_CTYPE_FOR_##ftype*)addr_out,	\
						 *(ATOM_CTYPE_FOR_##ftype*)buf,	\
						 *(ATOM_CTYPE_FOR_##ftype*)compare );	\
		break;								\
	}
#define atomic_case_compare_cplx(ftype, fop, absfun)				\
case ftype*FI_ATOMIC_OP_LAST + fop:						\
	{   if(result) {*(ATOM_CTYPE_FOR_##ftype*)result = *(ATOM_CTYPE_FOR_##ftype*)addr_in; }	\
		ATOM_FOR_CPLX_##fop(  *(ATOM_CTYPE_FOR_##ftype*)addr_in,	\
						 *(ATOM_CTYPE_FOR_##ftype*)addr_out,	\
						 *(ATOM_CTYPE_FOR_##ftype*)buf,	\
						 *(ATOM_CTYPE_FOR_##ftype*)compare,	\
						 absfun );			\
		break;								\
	}

#define atomic_int_ops(int_type) 			\
	atomic_case(int_type, FI_MIN)			\
	atomic_case(int_type, FI_MAX)			\
	atomic_case(int_type, FI_SUM)			\
	atomic_case(int_type, FI_PROD)			\
	atomic_case(int_type, FI_LOR)			\
	atomic_case(int_type, FI_LAND)			\
	atomic_case(int_type, FI_BOR)			\
	atomic_case(int_type, FI_BAND)			\
	atomic_case(int_type, FI_LXOR)			\
	atomic_case(int_type, FI_BXOR)			\
	atomic_case(int_type, FI_ATOMIC_READ)		\
	atomic_case(int_type, FI_ATOMIC_WRITE)		\
	atomic_case_compare(int_type, FI_CSWAP)		\
	atomic_case_compare(int_type, FI_CSWAP_NE)	\
	atomic_case_compare(int_type, FI_CSWAP_LE)	\
	atomic_case_compare(int_type, FI_CSWAP_LT)	\
	atomic_case_compare(int_type, FI_CSWAP_GE)	\
	atomic_case_compare(int_type, FI_CSWAP_GT)	\
	atomic_case_compare(int_type, FI_MSWAP)


#define atomic_real_float_ops(real_type)		\
	atomic_case(real_type, FI_MIN)			\
	atomic_case(real_type, FI_MAX)			\
	atomic_case(real_type, FI_SUM)			\
	atomic_case(real_type, FI_PROD)			\
	atomic_case(real_type, FI_LOR)			\
	atomic_case(real_type, FI_LAND)			\
	atomic_case(real_type, FI_LXOR)			\
	atomic_case(real_type, FI_ATOMIC_READ)		\
	atomic_case(real_type, FI_ATOMIC_WRITE)		\
	atomic_case_compare(real_type, FI_CSWAP)	\
	atomic_case_compare(real_type, FI_CSWAP_NE)	\
	atomic_case_compare(real_type, FI_CSWAP_LE)	\
	atomic_case_compare(real_type, FI_CSWAP_LT)	\
	atomic_case_compare(real_type, FI_CSWAP_GE)	\
	atomic_case_compare(real_type, FI_CSWAP_GT)

#define atomic_complex_float_ops(ctype, absfun)			\
	atomic_case_cplx(ctype, FI_MIN, absfun)			\
	atomic_case_cplx(ctype, FI_MAX, absfun)			\
	atomic_case(ctype, FI_SUM)				\
	atomic_case(ctype, FI_PROD)				\
	atomic_case(ctype, FI_LOR)				\
	atomic_case(ctype, FI_LAND)				\
	atomic_case(ctype, FI_LXOR)				\
	atomic_case(ctype, FI_ATOMIC_READ)			\
	atomic_case(ctype, FI_ATOMIC_WRITE)			\
	atomic_case_compare(ctype, FI_CSWAP)			\
	atomic_case_compare(ctype, FI_CSWAP_NE)			\
	atomic_case_compare_cplx(ctype, FI_CSWAP_LE, absfun)	\
	atomic_case_compare_cplx(ctype, FI_CSWAP_LT, absfun)	\
	atomic_case_compare_cplx(ctype, FI_CSWAP_GE, absfun)	\
	atomic_case_compare_cplx(ctype, FI_CSWAP_GT, absfun)


int perform_atomic_op(	enum fi_datatype dtype,
			enum fi_op op,
			void *addr_in,
			void *buf,
			void *addr_out,
			void *compare,
			void *result)
{
	switch(dtype*FI_ATOMIC_OP_LAST + op) {
		atomic_int_ops(FI_INT32)
		atomic_int_ops(FI_UINT32)
		atomic_int_ops(FI_INT8)
		atomic_int_ops(FI_UINT8)
		atomic_int_ops(FI_INT16)
		atomic_int_ops(FI_UINT16)
		atomic_int_ops(FI_INT64)
		atomic_int_ops(FI_UINT64)

		atomic_real_float_ops(FI_FLOAT)
		atomic_real_float_ops(FI_DOUBLE)
		atomic_real_float_ops(FI_LONG_DOUBLE)

		atomic_complex_float_ops(FI_FLOAT_COMPLEX, cabsf)
		atomic_complex_float_ops(FI_DOUBLE_COMPLEX, cabs)
		atomic_complex_float_ops(FI_LONG_DOUBLE_COMPLEX, cabsl)

		default:
			return -ENODATA;

	}
	return 0;
}

static int validation_input_value(enum fi_datatype datatype, int jrank, void *val) {
	switch(datatype) {
		case FI_INT8:
			*(int8_t*)val = (1+jrank)*10; break;
		case FI_INT16:
			*(int16_t*)val = (1+jrank)*10; break;
		case FI_INT32:
			*(int32_t*)val = (1+jrank)*10; break;
		case FI_INT64:
			*(int64_t*)val = (1+jrank)*10; break;
		case FI_UINT8:
			*(uint8_t*)val = (1+jrank)*10; break;
		case FI_UINT16:
			*(uint16_t*)val = (1+jrank)*10; break;
		case FI_UINT32:
			*(uint32_t*)val = (1+jrank)*10; break;
		case FI_UINT64:
			*(uint64_t*)val = (1+jrank)*10; break;
		case FI_FLOAT:
			*(float*)val = (1+jrank)*1.11; break;
		case FI_DOUBLE:
			*(double*)val = (1+jrank)*1.11; break;
		case FI_LONG_DOUBLE:
			*(long double*)val = (1+jrank)*1.11; break;
		case FI_FLOAT_COMPLEX:
			*(float complex*)val = CMPLXF( (1+jrank)*1.11, (1+jrank*-0.5) ); break;
		case FI_DOUBLE_COMPLEX:
			*(double complex*)val = CMPLX( (1+jrank)*1.11, (1+jrank*-0.5) ); break;
		case FI_LONG_DOUBLE_COMPLEX:
			*(long double complex*)val = CMPLXL( (1+jrank)*1.11, (1+jrank*-0.5) ); break;
		default:
			fprintf(stderr, "No initial value defined, cannot perform data validation "
					"on atomic operations using %s\n",
				fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE) );
			return -ENODATA;
	}
	return 0;
}

#define COMPARE_AS_TYPE(c_type, a, b) *(c_type*)(a) == *(c_type*)(b)
static int atom_binary_compare(enum fi_datatype dtype, void *a, void *b)
{
	switch (datatype_to_size(dtype))
	{
		case 1: return COMPARE_AS_TYPE(int8_t, a, b);
		case 2: return COMPARE_AS_TYPE(int16_t, a, b);
		case 4: return COMPARE_AS_TYPE(int32_t, a, b);
		case 8: return COMPARE_AS_TYPE(int64_t, a, b);
#ifdef HAVE___INT128
		case 16: return COMPARE_AS_TYPE(ofi_int128_t, a, b);
#endif
		default:
			fprintf(stderr, "No implementation of COMPARE_AS_TYPE for %s\n", fi_tostr(&dtype, FI_TYPE_ATOMIC_TYPE) );
		return 0;
	}
}

int atomic_data_validation_setup(enum fi_datatype datatype, int jrank, void *buf, size_t buf_size) {
	char set_value[16]; // fits maximum atom size of 128 bits.
	char set_buf[buf_size];
	int jatom;
	const size_t dtype_size = datatype_to_size(datatype);
	size_t natoms = (buf_size-dtype_size)/dtype_size + 1;
	int err;

	// get the value we wish to set the memory to.
	err = validation_input_value(datatype, jrank, set_value);
	if (err == -ENODATA) return 0;
	if (err) return err;



	// fill a system buffer with the value
	for (jatom=0; jatom < natoms; jatom++) {
		memcpy( set_buf + jatom*dtype_size, set_value, dtype_size );
	}

	// copy system buffer to hmem.
	err = ft_hmem_copy_to(opts.iface, opts.device, buf, set_buf, buf_size );
	return err;
}

#define PRINT_ADR_COMPARISON(dtype,fmt,ai,bi,ci,ao,ae) \
	fprintf(stderr, \
		"Initial Values: [local]addr=" fmt ", [remote]buf=" fmt ", [remote]compare=" fmt "\n" \
		"Observed Final Value: addr=" fmt "\n" \
		"Expected Final Value: addr=" fmt "\n", \
		*(ATOM_CTYPE_FOR_##dtype*)(ai), \
		*(ATOM_CTYPE_FOR_##dtype*)(bi), \
		*(ATOM_CTYPE_FOR_##dtype*)(ci), \
		*(ATOM_CTYPE_FOR_##dtype*)(ao), \
		*(ATOM_CTYPE_FOR_##dtype*)(ae) );
#define PRINT_RES_COMPARISON(dtype,fmt,ai,bi,ci,ro,re) \
	fprintf(stderr, \
		"Initial Values: [remote]addr=" fmt ", [local]buf=" fmt ", [local]compare=" fmt "\n" \
		"Observed Final Value: result=" fmt "\n" \
		"Expected Final Value: result=" fmt "\n", \
		*(ATOM_CTYPE_FOR_##dtype*)(ai), \
		*(ATOM_CTYPE_FOR_##dtype*)(bi), \
		*(ATOM_CTYPE_FOR_##dtype*)(ci), \
		*(ATOM_CTYPE_FOR_##dtype*)(ro), \
		*(ATOM_CTYPE_FOR_##dtype*)(re) )
static void print_failure_message(enum fi_datatype datatype,
	void *adr_in, void *buf_in, void *compare_in,
	void *adr_obs, void *res_obs,
	void *adr_expect, void *res_expect)
{
		switch (datatype) {
			case FI_INT8:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_INT8,"%d",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_INT8,"%d",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			case FI_UINT8:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_UINT8,"%u",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_UINT8,"%u",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			case FI_INT16:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_INT16,"%d",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_INT16,"%d",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			case FI_UINT16:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_UINT16,"%u",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_UINT16,"%u",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			case FI_INT32:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_INT32,"%d",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_INT32,"%d",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			case FI_UINT32:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_UINT32,"%u",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_UINT32,"%u",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			case FI_INT64:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_UINT64,"%ld",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_UINT64,"%ld",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			case FI_UINT64:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_UINT64,"%lu",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_UINT64,"%lu",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			case FI_FLOAT:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_FLOAT,"%f",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_FLOAT,"%f",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			case FI_DOUBLE:
				if (adr_obs) PRINT_ADR_COMPARISON(FI_DOUBLE,"%f",adr_in,buf_in,compare_in,adr_obs,adr_expect);
				if (res_obs) PRINT_RES_COMPARISON(FI_DOUBLE,"%f",adr_in,buf_in,compare_in,res_obs,res_expect);
				break;
			default:
				break;
		}
}

int atomic_data_validation_check(enum fi_datatype datatype, enum fi_op op, int jrank, void *addr, void *res, size_t buf_size, bool check_addr, bool check_result) {
	// these all fit the maximum atom size of 128 bits.
	const int MAX_ATOM_BYTES=16;
	char local_addr[MAX_ATOM_BYTES],            remote_addr[MAX_ATOM_BYTES];
	char local_buf[MAX_ATOM_BYTES],             remote_buf[MAX_ATOM_BYTES];
	char local_compare[MAX_ATOM_BYTES],         remote_compare[MAX_ATOM_BYTES];
	char expected_local_addr[MAX_ATOM_BYTES],   dummy_remote_addr[MAX_ATOM_BYTES];
	char expected_local_result[MAX_ATOM_BYTES];

	char local_addr_in_sysmem[buf_size];
	char local_result_in_sysmem[buf_size];
	size_t dtype_size = datatype_to_size(datatype);
	size_t natoms = (buf_size-dtype_size)/dtype_size + 1;
	int jatom;
	int err, addr_eq, res_eq, any_errors=0;
	int jrank_remote = (jrank+1)%2;


	// setup initial conditions so we can mock the test
	err  = validation_input_value(datatype, jrank, local_addr);
	err |= validation_input_value(datatype, jrank, local_buf);
	err |= validation_input_value(datatype, jrank, local_compare);
	err |= validation_input_value(datatype, jrank, expected_local_addr);
	err |= validation_input_value(datatype, jrank_remote, remote_addr);
	err |= validation_input_value(datatype, jrank_remote, remote_buf);
	err |= validation_input_value(datatype, jrank_remote, remote_compare);
	if (err == -ENODATA) goto nocheck;
	if (err) goto error;

	// mock the remote side performing operations on our local addr
	err  = perform_atomic_op(datatype, op, local_addr, remote_buf, expected_local_addr, remote_compare, NULL);
	// mock the local side performing operations on remote addr
	err |= perform_atomic_op(datatype, op, remote_addr, local_buf, dummy_remote_addr, local_compare, expected_local_result);
	if (err == -ENODATA) goto nocheck;
	if (err) goto error;

	err  = ft_hmem_copy_from(opts.iface, opts.device, local_addr_in_sysmem, addr, buf_size );
	err |= ft_hmem_copy_from(opts.iface, opts.device, local_result_in_sysmem, res, buf_size );
	if (err) goto error;

	for (jatom=0; jatom < natoms; jatom++) {
		addr_eq = true;
		res_eq = true;
		if (check_addr) {
			addr_eq = atom_binary_compare( datatype, expected_local_addr,
										   local_addr_in_sysmem + jatom*dtype_size);
		}
		if (!addr_eq) {
			fprintf( stderr, "FAILED: Remote atomic operation %s",fi_tostr(&op, FI_TYPE_ATOMIC_OP));
			fprintf(stderr, " on %s failed validation of addr at atom index %d.\n",
				fi_tostr(&datatype,    FI_TYPE_ATOMIC_TYPE),
				jatom );
			print_failure_message( datatype,
				local_addr, remote_buf, remote_compare,
				local_addr_in_sysmem + jatom*dtype_size, NULL,
				expected_local_addr, NULL);
		}
		if (check_result) {
			res_eq = atom_binary_compare( datatype, expected_local_result,
										  local_result_in_sysmem + jatom*dtype_size);
		}
		if (!res_eq) {
			fprintf( stderr, "FAILED: Local atomic operation %s",fi_tostr(&op, FI_TYPE_ATOMIC_OP));
			fprintf(stderr, " on %s failed validation of result at atom index %d.\n",
				fi_tostr(&datatype, FI_TYPE_ATOMIC_TYPE),
				jatom );
			print_failure_message( datatype,
				remote_addr, local_buf, local_compare,
				NULL, local_result_in_sysmem + jatom*dtype_size,
				NULL, expected_local_result);
		}
		if (!res_eq || !addr_eq) {
			any_errors = 1;
			break;
		}
	}
	atomic_dv_record(datatype, op, any_errors, true);
	return 0;

nocheck:
	atomic_dv_record(datatype, op, false, false);
	return 0;
error:
	atomic_dv_record(datatype, op, false, false);
	return err;


}
