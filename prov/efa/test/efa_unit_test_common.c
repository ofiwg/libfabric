#include "efa_unit_tests.h"

int efa_unit_test_resource_construct(struct efa_resource* resource)
{
	int ret;
	struct fi_av_attr av_attr = { 0 };

	ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, NULL, &resource->info);
	if (ret)
		return ret;

	ret = fi_fabric(resource->info->fabric_attr, &resource->fabric, NULL);
	if (ret) {
		fi_freeinfo(resource->info);
		return ret;
	}

	ret = fi_domain(resource->fabric, resource->info, &resource->domain, NULL);
	if (ret) {
		fi_close(&resource->fabric->fid);
		fi_freeinfo(resource->info);
		return ret;
	}

	ret = fi_av_open(resource->domain, &av_attr, &resource->av, NULL);
	if (ret) {
		fi_close(&resource->domain->fid);
		fi_close(&resource->fabric->fid);
		return ret;
	}

	return 0;
}

void efa_unit_test_resource_destroy(struct efa_resource* resource)
{
	fi_close(&resource->av->fid);
	fi_close(&resource->domain->fid);
	fi_close(&resource->fabric->fid);
	fi_freeinfo(resource->info);
}
