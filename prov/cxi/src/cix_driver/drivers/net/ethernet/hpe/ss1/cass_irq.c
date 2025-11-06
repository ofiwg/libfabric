// SPDX-License-Identifier: GPL-2.0
/* Copyright 2021 Hewlett Packard Enterprise Development LP */

/* Cassini interrupt management */

#include <linux/interrupt.h>
#include <linux/pci.h>
#include <linux/types.h>

#include "cass_core.h"

/*
 * cass_irq_int_handler() - Completion interrupt handler
 */
static irqreturn_t cass_irq_int_handler(int irq, void *nh)
{
	atomic_notifier_call_chain(nh, 0, NULL);

	return IRQ_HANDLED;
}

/*
 * cass_comp_irq_attach() - Attach to a completion interrupt
 *
 * @hw: Cassini device
 * @mask: CPU mask used as an affinity hint to select IRQ vector
 * @nb: Notifier block to be called on interrupt
 *
 * @return Interrupt is returned index on success. A negative errno is returned
 * on error.
 */
struct cass_irq *cass_comp_irq_attach(struct cass_dev *hw,
				      const struct cpumask *mask,
				      struct notifier_block *nb)
{
	struct cass_irq *best_irq = NULL;
	int cpu;
	int rc;
	int i;

	for_each_cpu(cpu, mask) {
		for (i = 0; i < hw->num_comp_irqs; i++) {
			struct cass_irq *irq = &hw->comp_irqs[i];

			if (!cpumask_test_cpu(cpu, &irq->mask))
				continue;

			if (atomic_read(&irq->refcount) == 0) {
				best_irq = irq;
				goto do_register;
			}

			if (!best_irq || atomic_read(&best_irq->refcount) >
					 atomic_read(&irq->refcount))
				best_irq = irq;
		}
	}

	/* If no match, select a random IRQ vector. */
	if (!best_irq) {
		i = get_random_u32() % hw->num_comp_irqs;
		best_irq = &hw->comp_irqs[i];
	}

do_register:
	rc = atomic_notifier_chain_register(&best_irq->nh, nb);
	if (rc < 0)
		return ERR_PTR(rc);

	mutex_lock(&best_irq->lock);

	if (atomic_read(&best_irq->refcount) == 0) {
		rc = request_irq(best_irq->vec, cass_irq_int_handler, 0,
				 best_irq->irq_name, &best_irq->nh);
		if (rc) {
			mutex_unlock(&best_irq->lock);

			cxidev_err(&hw->cdev, "request_irq failed at index %d: %d\n",
				   i, rc);

			atomic_notifier_chain_unregister(&best_irq->nh, nb);

			return ERR_PTR(rc);
		}

		rc = irq_set_affinity_hint(best_irq->vec, &best_irq->mask);
		if (rc)
			cxidev_err(&hw->cdev, "irq_set_affinity_hint failed: %d\n",
				   rc);
	}

	atomic_inc(&best_irq->refcount);

	mutex_unlock(&best_irq->lock);

	return best_irq;
}

/*
 * cass_comp_irq_detach() - Detach from a completion interrupt
 *
 * @hw: Cassini device
 * @irq: An interrupt returned from cass_comp_irq_attach()
 * @nb: The notifier block to detach
 */
void cass_comp_irq_detach(struct cass_dev *hw, struct cass_irq *irq,
			  struct notifier_block *nb)
{
	atomic_notifier_chain_unregister(&irq->nh, nb);

	mutex_lock(&irq->lock);

	if (atomic_dec_return(&irq->refcount) == 0) {
		irq_set_affinity_hint(irq->vec, NULL);
		free_irq(irq->vec, &irq->nh);
	} else {
		synchronize_irq(irq->vec);
	}

	mutex_unlock(&irq->lock);
}

/*
 * cass_irq_init() - Initialize interrupts
 */
int cass_irq_init(struct cass_dev *hw)
{
	int rc;
	int i;
	int nvecs_res = 0;
	int nvecs_min = 0;
	int nvecs_max = 0;
	struct cass_irq *irq;
	struct pci_dev *pdev = hw->cdev.pdev;
	int numa = dev_to_node(&pdev->dev);

	if (!hw->with_vf_support)
		return 0;

	/* Initialize the MSI vectors table */
	if (hw->cdev.is_physfn) {
		/* Reserve the special IRQ numbers */
		nvecs_res += C_FIRST_AVAIL_MSIX_INT;
	} else {
		/* Reserve IRQ number 0 for communicating with the PF. */
		nvecs_res += 1;
	}

	/* Reserve IRQs for ATU */
	hw->atu_cq_vec = nvecs_res++;
	hw->atu_pri_vec = nvecs_res++;

	/* Get as many IRQs as possible, but not too many as
	 * request_irq() will eventually fail.
	 */
	nvecs_min = nvecs_res + 1;
	if (HW_PLATFORM_NETSIM(hw))
		nvecs_max = 128;
	else
		nvecs_max = pci_msix_vec_count(pdev);

	/* Enable MSIX. */
	rc = pci_alloc_irq_vectors(pdev, nvecs_min, nvecs_max, PCI_IRQ_MSIX);
	if (rc < 0)
		return rc;

	hw->num_irqs = rc;
	hw->num_comp_irqs = hw->num_irqs - nvecs_res;

	/* Use remaining IRQs for completions */
	hw->comp_irqs = kcalloc(hw->num_comp_irqs, sizeof(struct cass_irq),
				GFP_KERNEL);
	if (!hw->comp_irqs) {
		rc = -ENOMEM;
		goto free_irq_vecs;
	}

	for (i = 0; i < hw->num_comp_irqs; i++) {
		irq = &hw->comp_irqs[i];
		irq->idx = nvecs_res + i;
		ATOMIC_INIT_NOTIFIER_HEAD(&irq->nh);
		atomic_set(&irq->refcount, 0);
		irq->vec = pci_irq_vector(pdev, irq->idx);
		mutex_init(&irq->lock);

		scnprintf(irq->irq_name, sizeof(irq->irq_name),
			  "%s_comp%d", hw->cdev.name, i);
		cpumask_set_cpu(cpumask_local_spread(i, numa), &irq->mask);
	}

	return 0;

free_irq_vecs:
	pci_free_irq_vectors(pdev);

	return rc;
}

/*
 * cass_irq_fini() - Finalize interrupts
 */
void cass_irq_fini(struct cass_dev *hw)
{
	if (!hw->with_vf_support)
		return;

	kfree(hw->comp_irqs);

	pci_free_irq_vectors(hw->cdev.pdev);
}
