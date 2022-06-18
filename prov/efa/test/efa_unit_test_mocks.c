#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "efa.h"

void __real_rxr_pkt_handle_send_completion(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry);

void __wrap_rxr_pkt_handle_send_completion(struct rxr_ep *ep, struct rxr_pkt_entry *pkt_entry)
{
	check_expected(ep);
	check_expected(pkt_entry);

	if ((int)mock() == 4242) {
		__real_rxr_pkt_handle_send_completion(ep, pkt_entry);
	}
}

void __real_rxr_pkt_handle_recv_completion(struct rxr_ep *ep,
										   struct rxr_pkt_entry *pkt_entry,
										   enum rxr_lower_ep_type lower_ep_type);

void __wrap_rxr_pkt_handle_recv_completion(struct rxr_ep *ep,
										   struct rxr_pkt_entry *pkt_entry,
										   enum rxr_lower_ep_type lower_ep_type)
{
	check_expected(ep);
	check_expected(pkt_entry);
	check_expected(lower_ep_type);

	if ((int)mock() == 4242) {
		__real_rxr_pkt_handle_recv_completion(ep, pkt_entry, lower_ep_type);
	}
};

void __real_rxr_pkt_handle_send_error(struct rxr_ep *ep,
			       struct rxr_pkt_entry *pkt_entry,
			       int err, int prov_errno);

void __wrap_rxr_pkt_handle_send_error(struct rxr_ep *ep,
									  struct rxr_pkt_entry *pkt_entry,
									  int err, int prov_errno)
{
	check_expected(ep);
	check_expected(pkt_entry);
	check_expected(err);
	check_expected(prov_errno);

	if ((int)mock() == 4242) {
		__real_rxr_pkt_handle_send_error(ep, pkt_entry, err, prov_errno);
	}
}

void __real_rxr_pkt_handle_recv_error(struct rxr_ep *ep,
			       struct rxr_pkt_entry *pkt_entry,
			       int err, int prov_errno);

void __wrap_rxr_pkt_handle_recv_error(struct rxr_ep *ep,
			       struct rxr_pkt_entry *pkt_entry,
			       int err, int prov_errno)
{
	check_expected(ep);
	check_expected(pkt_entry);
	check_expected(err);
	check_expected(prov_errno);

	if ((int)mock() == 4242) {
		__real_rxr_pkt_handle_recv_error(ep, pkt_entry, err, prov_errno);
	}
}
