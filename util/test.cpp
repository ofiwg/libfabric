#include <rdma/fabric.hpp>
#include <rdma/domain.hpp>

#include <stdio.h>

int main (int argc, char *argv[]){

	struct fi_info *fi, *hints = fi_allocinfo();
	hints->mode = ~0;
	hints->domain_attr->mode = ~0;
	hints->domain_attr->mr_mode = ~0;
	hints->fabric_attr->prov_name = (char*)"verbs";

	fi_getinfo(FI_VERSION(1,5), NULL, 0, 0, hints, &fi);

	try {
		fi::fabric f(fi->fabric_attr, NULL);
		fi::domain d = f.domain(fi, NULL);
		fi::pep p = f.passive_ep(fi, NULL);
		fi::ep ep = d.endpoint(fi, NULL);
		fi::pep p2 = p;
		if (p.listen())
			printf("failed to listen\n");
		else
			printf("listening\n");
	} catch (fi::fi_error &ex) {
		printf("caught exception: %s\n", ex.what());
	}

	return 0;
}
