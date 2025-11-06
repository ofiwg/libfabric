// SPDX-License-Identifier: GPL-2.0
/* Copyright 2021 Hewlett Packard Enterprise Development LP */


/*
 * ODP decoupling in software to work around Cassini ERRATA-3260
 *
 * Notes:
 *
 * 1. We will enter ODP decoupling when an invalidation completion wait
 *    times out.
 * 2. Find out the ID of the previous epoch. The completion wait command
 *    triggers the switch between epochs. We need to know which one is blocked.
 *    Read C_ATU_STS_AT_EPOCH. The previous epoch x = !CURRENT_EPOCH
 * 3. If IXE_EPOCH*x*CNTR_NE0 is zero. The completion wait just finished.
 *    Return from decoupling procedure and continue ODP processing. (return)
 * 4. If IXE_EPOCH*x*_CNTR_NE0 is one, we need to find out if this epoch
 *    counter is stuck or continues to decrement. Read repeatedly to see if
 *    EPOCH*x* is moving. The interval between reads should be around 32
 *    microseconds.
 * 5. If EPOCH*x* returns to zero, return from decoupling procedure. (return)
 * 6. If EPOCH*x* does not change between two reads, we determine that WRQ is
 *    stuck with ODP at the head of the queue.
 * 6.1  Get the previous Inbound Epoch. z = !C_ATU_STS_IB_EPOCH.CURRENT_EPOCH
 * 6.2  Check C_ATU_STS_IB_EPOCH_CNTR.EPOCH*z*
 *      If it is zero, we are not in the active Inbound wait and ready to
 *      issue a new InboundWait (step 6.3)
 *      If it is not zero, InboundWait is in progress. We need to wait for
 *      this counter to get to zero before we issue a new one (step 6.3).
 *      This counter may stop decrementing before it gets to zero. If it
 *      happens, we do not need to issue a new InboundWait and should
 *      proceed to step #7.
 * 6.3  Issue an InboundWait and update z = !C_ATU_STS_IB_EPOCH.CURRENT_EPOCH
 * 6.4  Monitor C_ATU_STS_IB_EPOCH_CNTR.EPOCH*z*
 *      If it returns to zero, exit this procedure. Check
 *      C_ATU_STS_IXE_AT_EPOCH_CNTR.EPOCHx. We expect the translation epoch
 *      to be zero as well so no need for HW decoupling. If the translation
 *      epoch counter is not zero, return a critical error.
 *      If the counter stops decrementing, proceed to step #7.
 * 7. To avoid Cassini ERRATA-3241, make sure EE and OXE epochs are at zero. Read
 *    C_ATU_STS_AT_EPOCH to check if OXE_EPOCH*x*_CNTR_NE0 and
 *    EE_EPOCH*x*_CNTR_NE0 are zero. If that is not the case, start pulling
 *    C_ATU_STS_OXE_AT_EPOCH_CNTR and C_ATU_STS_EE_AT_EPOCH_CNTR, checking
 *    EPOCH*x*, at 100 microseconds intervals. These counters should decrement
 *    to zero. Detecting one of the counters being stuck indicates an
 *    unexpected problem; print a critical error.
 * 8. At this point, it is safe to enable HW decoupling
 *    (C_ATU_CFG_ODP_DECOUPLE.ENABLE=1)
 * 9. Continue reading C_ATU_STS_IXE_AT_EPOCH_CNTR, waiting for EPOCH*x* to
 *    return to zero. If it does not happen for 100 ms, we have a problem,
 *    print a critical error to get the node rebooted.
 * 10.Once EPOCH*x* values for all three clients are zero, disable HW
 *    decoupling (C_ATU_CFG_ODP_DECOUPLE.ENABLE=0) continue processing
 *    invalidations. Return from decoupling procedure and continue ODP
 *    processing. (return)
 */

#include <linux/iopoll.h>
#include "cass_core.h"

#define IXE_ECNTR C_ATU_STS_IXE_AT_EPOCH_CNTR
#define IB_ECNTR C_ATU_STS_IB_EPOCH_CNTR

static int ibw_epoch_wait_us = 50;
module_param(ibw_epoch_wait_us, int, 0644);
MODULE_PARM_DESC(ibw_epoch_wait_us,
		 "Inbound epoch counter decrement wait time");

static int ixe_epoch_wait_us = 32;
module_param(ixe_epoch_wait_us, int, 0644);
MODULE_PARM_DESC(ixe_epoch_wait_us,
		 "Initial IXE epoch counter decrement wait time");

static int epoch_wait_us = 100;
module_param(epoch_wait_us, int, 0644);
MODULE_PARM_DESC(epoch_wait_us, "Epoch counter decrement wait time");

static void enable_hw_decoupling(struct cass_dev *hw, bool enable)
{
	union c1_atu_cfg_odp_decouple odp_dcpl = {
		.flush_req_delay = ATU_FLUSH_REQ_DELAY,
		.enable = enable
	};

	cass_write(hw, C1_ATU_CFG_ODP_DECOUPLE, &odp_dcpl, sizeof(odp_dcpl));
}

/* read/clear file operations */
static int decouple_stats_read(struct seq_file *s, void *unused)
{
	int i;
	long bin;
	struct timespec64 time;
	struct cass_dev *hw = s->private;

	seq_printf(s, "completion_wait:%d\n",
		   atomic_read(&hw->dcpl_comp_wait));
	seq_printf(s, "md_clear_inval:%d\n",
		   atomic_read(&hw->dcpl_md_clear_inval));
	seq_printf(s, "nta_mn_inval:%d\n",
		   atomic_read(&hw->dcpl_nta_mn_inval));
	seq_printf(s, "ats_mn_inval:%d\n",
		   atomic_read(&hw->dcpl_ats_mn_inval));
	seq_printf(s, "ee_cntr_dec_0:%d\n",
		   atomic_read(&hw->dcpl_ee_cntr_dec_0));
	seq_printf(s, "ee_cntr_stuck:%d\n",
		   atomic_read(&hw->dcpl_ee_cntr_stuck));
	seq_printf(s, "oxe_cntr_dec_0:%d\n",
		   atomic_read(&hw->dcpl_oxe_cntr_dec_0));
	seq_printf(s, "oxe_cntr_stuck:%d\n",
		   atomic_read(&hw->dcpl_oxe_cntr_stuck));
	seq_printf(s, "ixe_cntr_0:%d\n",
		   atomic_read(&hw->dcpl_ixe_cntr_0));
	seq_printf(s, "ixe_cntr_dec_0:%d\n",
		   atomic_read(&hw->dcpl_ixe_cntr_dec_0));
	seq_printf(s, "ixe_cntr_stuck:%d\n",
		   atomic_read(&hw->dcpl_ixe_cntr_stuck));
	seq_printf(s, "ibw_cntr_dec_0_62:%d\n",
		   atomic_read(&hw->dcpl_ibw_cntr_dec_0_62));
	seq_printf(s, "ibw_cntr_dec_0_62_count:%d\n",
		   atomic_read(&hw->dcpl_ibw_cntr_dec_0_62_count));
	seq_printf(s, "ibw_cntr_dec_0_64:%d\n",
		   atomic_read(&hw->dcpl_ibw_cntr_dec_0_64));
	seq_printf(s, "ibw_cntr_dec_0_64_count:%d\n",
		   atomic_read(&hw->dcpl_ibw_cntr_dec_0_64_count));
	seq_printf(s, "step7:%d\n",
		   atomic_read(&hw->dcpl_step7));
	seq_printf(s, "ibw_active_stuck:%d\n",
		   atomic_read(&hw->dcpl_ibw_active_stuck));
	seq_printf(s, "ibw_idle_wait:%d\n",
		   atomic_read(&hw->dcpl_ibw_idle_wait));
	seq_printf(s, "ibw_cntr_is_0:%d\n",
		   atomic_read(&hw->dcpl_ibw_cntr_is_0));
	seq_printf(s, "ibw_issued:%d\n",
		   atomic_read(&hw->dcpl_ibw_issued));
	seq_printf(s, "entered:%d\n",
		   atomic_read(&hw->dcpl_entered));
	seq_printf(s, "success:%d\n",
		   atomic_read(&hw->dcpl_success));

	time = ktime_to_timespec64(hw->dcpl_max_time);
	seq_printf(s, "dcpl_max_time:%lld.%06lds\n",
		       time.tv_sec,
		       time.tv_nsec / NSEC_PER_USEC);

	for (i = 0, bin = FIRST_BIN; i < DBINS; i++, bin <<= 1) {
		time = ktime_to_timespec64(bin);
		seq_printf(s, "dcpl_time_bin:%lld.%06lds:%d\n",
			   time.tv_sec, time.tv_nsec / NSEC_PER_USEC,
			   hw->dcpl_time[i]);
	}

	time = ktime_to_timespec64(hw->pri_max_md_time);
	seq_printf(s, "pri_max_md_time:%lld.%06lds\n",
		       time.tv_sec,
		       time.tv_nsec / NSEC_PER_USEC);

	for (i = 0, bin = FIRST_MDBIN; i < MDBINS; i++, bin <<= 1) {
		time = ktime_to_timespec64(bin);
		seq_printf(s, "pri_md_time_bin:%lld.%06lds:%d\n",
			   time.tv_sec, time.tv_nsec / NSEC_PER_USEC,
			   hw->pri_md_time[i]);
	}

	time = ktime_to_timespec64(hw->pri_max_fault_time);
	seq_printf(s, "pri_max_fault_time:%lld.%06lds\n",
		       time.tv_sec,
		       time.tv_nsec / NSEC_PER_USEC);

	for (i = 0, bin = FIRST_FBIN; i < FBINS; i++, bin <<= 1) {
		time = ktime_to_timespec64(bin);
		seq_printf(s, "pri_fault_time_bin:%lld.%06lds:%d\n",
			   time.tv_sec, time.tv_nsec / NSEC_PER_USEC,
			   hw->pri_fault_time[i]);
	}

	return 0;
}

static ssize_t decouple_stats_write(struct file *f, const char __user *buf,
				  size_t size, loff_t *pos)
{
	int i;
	struct cass_dev *hw = file_inode(f)->i_private;

	atomic_set(&hw->dcpl_md_clear_inval, 0);
	atomic_set(&hw->dcpl_nta_mn_inval, 0);
	atomic_set(&hw->dcpl_ats_mn_inval, 0);
	atomic_set(&hw->dcpl_comp_wait, 0);
	atomic_set(&hw->dcpl_entered, 0);
	atomic_set(&hw->dcpl_ixe_cntr_0, 0);
	atomic_set(&hw->dcpl_ixe_cntr_dec_0, 0);
	atomic_set(&hw->dcpl_ee_cntr_dec_0, 0);
	atomic_set(&hw->dcpl_ee_cntr_stuck, 0);
	atomic_set(&hw->dcpl_oxe_cntr_dec_0, 0);
	atomic_set(&hw->dcpl_oxe_cntr_stuck, 0);
	atomic_set(&hw->dcpl_ixe_cntr_stuck, 0);
	atomic_set(&hw->dcpl_success, 0);
	atomic_set(&hw->dcpl_ibw_cntr_dec_0_62, 0);
	atomic_set(&hw->dcpl_ibw_cntr_dec_0_62_count, 0);
	atomic_set(&hw->dcpl_ibw_cntr_dec_0_64, 0);
	atomic_set(&hw->dcpl_ibw_cntr_dec_0_64_count, 0);
	atomic_set(&hw->dcpl_step7, 0);
	atomic_set(&hw->dcpl_ibw_active_stuck, 0);
	atomic_set(&hw->dcpl_ibw_idle_wait, 0);
	atomic_set(&hw->dcpl_ibw_cntr_is_0, 0);
	atomic_set(&hw->dcpl_ibw_issued, 0);

	hw->dcpl_max_time = 0;
	for (i = 0; i < DBINS; i++)
		hw->dcpl_time[i] = 0;

	hw->pri_max_md_time = 0;
	for (i = 0; i < MDBINS; i++)
		hw->pri_md_time[i] = 0;
	hw->pri_max_fault_time = 0;
	for (i = 0; i < FBINS; i++)
		hw->pri_fault_time[i] = 0;

	return size;
}

static int decouple_stats_read_open(struct inode *inode, struct file *file)
{
	return single_open(file, decouple_stats_read, inode->i_private);
}

const struct file_operations decouple_stats_fops = {
	.owner = THIS_MODULE,
	.open = decouple_stats_read_open,
	.read = seq_read,
	.write = decouple_stats_write,
	.llseek  = seq_lseek,
	.release = single_release,
};

static ssize_t sw_dcpl_read(struct file *f, char __user *buf, size_t size,
			    loff_t *pos)
{
	char b[3];

	b[0] = odp_sw_decouple ? 'Y' : 'N';
	b[1] = '\n';
	b[2] = '\0';

	return simple_read_from_buffer(buf, size, pos, b, 3);
}

static ssize_t sw_dcpl_write(struct file *f, const char __user *buf,
			     size_t size, loff_t *pos)
{
	int ret;
	bool user_value;
	struct cass_dev *hw = file_inode(f)->i_private;

	ret = kstrtobool_from_user(buf, size, &user_value);
	if (ret)
		return ret;

	odp_sw_decouple = user_value;
	enable_hw_decoupling(hw, !odp_sw_decouple);

	return size;
}

const struct file_operations sw_decouple_fops = {
	.owner = THIS_MODULE,
	.read = sw_dcpl_read,
	.write = sw_dcpl_write,
	.llseek = default_llseek
};


static void add_dcpl_time_to_bin(struct cass_dev *hw, ktime_t time)
{
	int i;
	int bin = FIRST_BIN;

	if (ktime_compare(hw->dcpl_max_time, time) < 0)
		hw->dcpl_max_time = time;

	for (i = 0; i < DBINS; i++, bin <<= 1) {
		if (time < bin) {
			hw->dcpl_time[i]++;
			break;
		}
	}
}

/**
 * cass_wait_epoch_cntr_dec_zero()
 *
 * Wait for one of the c_atu_sts_*_at_epoch_cntr counters to decrement to 0.
 * The counters all use the same struct.
 *
 * @hw: the device
 * @csr: Address of CSR to poll
 * @sleep_us: Sleep time in microseconds
 * @epoch: Which counter (0 or 1) to poll
 * @loop_count: How many times through the loop. If the count is 0 the first
 *              time, the count will be zero.
 * Return: 0 on success, -1 if not decrementing
 */
static int cass_wait_epoch_cntr_dec_zero(struct cass_dev *hw, u64 csr,
					 ulong sleep_us, int epoch,
					 int *loop_count)
{
	int i;
	int ret = 0;
	u64 count;
	u64 last_count = 0xffff;
	union c_atu_sts_ixe_at_epoch_cntr ecntr;

	for (i = 0; last_count > 0; i++) {
		cass_read(hw, csr, &ecntr, sizeof(ecntr));
		count = epoch ? ecntr.epoch1 : ecntr.epoch0;

		if (!count)
			break;

		if (count >= last_count) {
			ret = -1;
			break;
		}

		last_count = count;
		usleep_range(sleep_us, sleep_us * 2);
	};

	*loop_count = i;

	return ret;
}

/**
 * ixe_epoch_cntr_0() - Check if IXE epoch counter is zero
 *
 * @epoch_sts: Epoch status
 * @epoch: The current epoch (false for epoch 0, true for epoch 1)
 * Return: true if zero, false if not zero
 */
static bool ixe_epoch_cntr_0(union c_atu_sts_at_epoch *epoch_sts, bool epoch)
{
	if (!epoch) {
		if (!epoch_sts->ixe_epoch0_cntr_ne0)
			return true;
	} else {
		if (!epoch_sts->ixe_epoch1_cntr_ne0)
			return true;
	}

	return false;
}

/**
 * ibw_epoch_cntr_is_0() - Check if IBW epoch counter is zero
 *
 * @hw: the device
 * @epoch_ret: Return the current_epoch
 * Return: true if zero, false if not zero
 */
static bool ibw_epoch_cntr_is_0(struct cass_dev *hw, bool *epoch_ret)
{
	bool epoch;
	union c_atu_sts_ib_epoch epoch_sts;

	cass_read(hw, C_ATU_STS_IB_EPOCH, &epoch_sts, sizeof(epoch_sts));
	epoch = !epoch_sts.current_epoch;

	*epoch_ret = epoch;

	if (!epoch) {
		if (!epoch_sts.epoch0_cntr_ne0)
			return true;
	} else {
		if (!epoch_sts.epoch1_cntr_ne0)
			return true;
	}

	return false;
}

/**
 * ibw_epoch_cntr_is_stuck() - Check if IBW counter is not decrementing
 *                             Steps 6.1 and 6.2
 *
 * @hw: the device
 * Return: true if stuck, false if 0 or returned to 0
 */
static bool ibw_epoch_cntr_is_stuck(struct cass_dev *hw)
{
	int ret;
	int count;
	bool ibw_epoch;

	/* Step 6.1 - Get the previous Inbound Epoch */
	ret = ibw_epoch_cntr_is_0(hw, &ibw_epoch);
	if (ret)
		return false;

	/* Step 6.2 - Wait for inbound epoch counter to decrement to zero. */
	ret = cass_wait_epoch_cntr_dec_zero(hw, IB_ECNTR, ibw_epoch_wait_us,
					    ibw_epoch, &count);
	if (!ret) {
		atomic_inc(&hw->dcpl_ibw_cntr_dec_0_62);
		atomic_add(count, &hw->dcpl_ibw_cntr_dec_0_62_count);
		return false;
	}

	return true;
}

/**
 * cass_odp_decouple() - handles spurious translation request due to
 * bug
 *
 * WORKAROUND for Cassini ERRATA-3260
 * In specific conditions, writes can be acked with a spurious pending
 * translation request remaining. If not addressed with this function,
 * then data corruption could occur.
 *
 * @hw: the device
 * Return: 0 on success, negative value on failure
 */
int cass_odp_decouple(struct cass_dev *hw)
{
	int ret = 0;
	int count;
	bool ibw_epoch;
	bool blocked_epoch;
	union c_atu_sts_at_epoch at_epoch_sts;
	union c_atu_sts_ixe_at_epoch_cntr ecntr;
	struct cass_atu_cq *cq = &hw->atu_cq;
	ktime_t start = ktime_get_raw();
	struct sts_idle ibw_idle = {
		.hw = hw,
		.ib_wait = true
	};

	/* Step 1 */
	atomic_inc(&hw->dcpl_entered);

	/* Step 2 */
	cass_read(hw, C_ATU_STS_AT_EPOCH, &at_epoch_sts, sizeof(at_epoch_sts));
	blocked_epoch = !at_epoch_sts.current_epoch;

	/* Step 3 - Check either IXE epoch counter 0 or 1 depending on the
	 * current epoch.
	 */
	if (ixe_epoch_cntr_0(&at_epoch_sts, blocked_epoch)) {
		atomic_inc(&hw->dcpl_ixe_cntr_0);
		goto done;
	}

	/* Step 4/5 - Check if C_ATU_STS_IXE_AT_EPOCH_CNTR is stuck or returns
	 * to 0.
	 */
	ret = cass_wait_epoch_cntr_dec_zero(hw, IXE_ECNTR, ixe_epoch_wait_us,
					    blocked_epoch, &count);
	if (!ret) {
		atomic_inc(&hw->dcpl_ixe_cntr_dec_0);
		goto done;
	}

	/* This function may have completed with IBW active so check if it
	 * still is. Also, inbound wait leaves the inbound_wait_idle inactive
	 * for some time after it has completed so we need to wait for idle
	 * before issuing a new inbound wait.
	 * If IBW is not idle, check if the IBW epoch counter is stuck.
	 */
	while (!cass_sts_idle(&ibw_idle)) {
		if (ibw_epoch_cntr_is_stuck(hw)) {
			atomic_inc(&hw->dcpl_ibw_active_stuck);
			goto step_7;
		}
		usleep_range(10, 20);
	}

	/* Determine if another process has issued an inbound wait since the
	 * lock will be held in that case.
	 * If we don't get the lock, then cxi_inbound_wait() is in progress.
	 * If IBW is idle, the c_atu_cfg_inbound_wait may not have been
	 * written yet or it is just completing and the lock will be released
	 * soon. We will wait for a not idle status before we can continue to
	 * step 6.1.
	 */
	while (!mutex_trylock(&cq->atu_ib_mutex)) {
		if (cass_sts_idle(&ibw_idle)) {
			atomic_inc(&hw->dcpl_ibw_idle_wait);
			usleep_range(10, 20);
			continue;
		}

		/* Step 6.1/6.2 */
		ret = ibw_epoch_cntr_is_stuck(hw);
		if (ret)
			goto step_7;

		atomic_inc(&hw->dcpl_ibw_cntr_is_0);
	}

	/* Check for idle after we have the lock */
	if (!cass_sts_idle(&ibw_idle)) {
		if (ibw_epoch_cntr_is_stuck(hw)) {
			mutex_unlock(&cq->atu_ib_mutex);
			goto step_7;
		}
	}

	/* Step 6.3 - Issue inbound wait and update current epoch */
	atomic_inc(&hw->dcpl_ibw_issued);
	ret = cass_inbound_wait(hw, false);
	if (ret) {
		cxidev_crit_once(&hw->cdev, "IBW failed %d\n", ret);
		mutex_unlock(&cq->atu_ib_mutex);
		goto done;
	}

	/* get the new epoch */
	ibw_epoch_cntr_is_0(hw, &ibw_epoch);
	mutex_unlock(&cq->atu_ib_mutex);

	/* 6.4  Monitor inbound epoch counter */
	ret = cass_wait_epoch_cntr_dec_zero(hw, IB_ECNTR, ibw_epoch_wait_us,
					    ibw_epoch, &count);
	if (!ret) {
		atomic_inc(&hw->dcpl_ibw_cntr_dec_0_64);
		atomic_add(count, &hw->dcpl_ibw_cntr_dec_0_64_count);

		cass_read(hw, C_ATU_STS_AT_EPOCH, &at_epoch_sts,
			  sizeof(at_epoch_sts));
		if (blocked_epoch != !at_epoch_sts.current_epoch)
			cxidev_crit_once(&hw->cdev, "IXE epoch changed\n");

		if (!ixe_epoch_cntr_0(&at_epoch_sts, blocked_epoch))
			cxidev_crit_once(&hw->cdev, "IXE epoch cntr should be 0\n");

		goto done;
	}

step_7:
	/* Step 7 - Check if C_ATU_STS_EE/OXE_AT_EPOCH_CNTR are at 0. */
	atomic_inc(&hw->dcpl_step7);

	cass_read(hw, C_ATU_STS_AT_EPOCH, &at_epoch_sts, sizeof(at_epoch_sts));
	if ((!blocked_epoch && at_epoch_sts.ee_epoch0_cntr_ne0) ||
			(blocked_epoch && at_epoch_sts.ee_epoch1_cntr_ne0)) {
		atomic_inc(&hw->dcpl_ee_cntr_dec_0);
		ret = cass_wait_epoch_cntr_dec_zero(hw,
				C_ATU_STS_EE_AT_EPOCH_CNTR, epoch_wait_us,
				blocked_epoch, &count);
		if (ret) {
			cxidev_crit_once(&hw->cdev,
				"EE_AT_EPOCH_CNTR did not return to 0\n");
			atomic_inc(&hw->dcpl_ee_cntr_stuck);
			ret = -1;
			goto done;
		}
	}

	if ((!blocked_epoch && at_epoch_sts.oxe_epoch0_cntr_ne0) ||
			(blocked_epoch && at_epoch_sts.oxe_epoch1_cntr_ne0)) {
		atomic_inc(&hw->dcpl_oxe_cntr_dec_0);
		ret = cass_wait_epoch_cntr_dec_zero(hw,
				C_ATU_STS_OXE_AT_EPOCH_CNTR,
				epoch_wait_us, blocked_epoch, &count);
		if (!ret) {
			cxidev_crit_once(&hw->cdev,
				"OXE_AT_EPOCH_CNTR did not return to 0\n");
			atomic_inc(&hw->dcpl_oxe_cntr_stuck);
			ret = -1;
			goto done;
		}
	}

	/* Step 8 - Now we can enable hw decoupling */
	enable_hw_decoupling(hw, true);

	/* Step 9 - Wait for IXE epoch cntr to return to 0 */
	ret = cass_wait_epoch_cntr_dec_zero(hw, IXE_ECNTR, epoch_wait_us,
					    blocked_epoch, &count);
	if (!ret) {
		atomic_inc(&hw->dcpl_success);
	} else {
		cass_read(hw, IXE_ECNTR, &ecntr,
			  sizeof(ecntr));
		cxidev_crit_once(&hw->cdev,
				 "IXE EPOCH counter not 0 with decoupling enabled epoch:%d epoch0:%x epoch1:%x\n",
				 blocked_epoch, ecntr.epoch0, ecntr.epoch1);
		ret = -1;
		atomic_inc(&hw->dcpl_ixe_cntr_stuck);
	}

	/* Step 10 - Disable hw decoupling */
	enable_hw_decoupling(hw, false);

done:
	add_dcpl_time_to_bin(hw, ktime_get_raw() - start);

	return ret;
}
