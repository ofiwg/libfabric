
int atomic_data_validation_print_summary();
int atomic_data_validation_setup(enum fi_datatype datatype, int jrank, void * buf, size_t buf_size);
int atomic_data_validation_check(enum fi_datatype datatype, enum fi_op op, int jrank, void *addr, void *res, size_t buf_size, bool check_addr, bool check_result);
int ft_sync_for_validation();
