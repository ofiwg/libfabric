// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019 Hewlett Packard Enterprise Development LP */

/* Cassini Precision Time Protocol support */

#include <linux/ptp_clock_kernel.h>

#include "cass_core.h"

#define RTC_NS_WIDTH 30
#define RTC_MAX_ADJ (RTC_NS_WIDTH / 2)
#define RTC_FNS_WIDTH 32
#define RTC_MAX_FNS (1ULL << RTC_FNS_WIDTH)

static void write_rtc(struct cass_dev *hw, const struct timespec64 *ts)
{
	union c_mb_cfg_rtc_time rtc;
	const union c_mb_cfg_rtc rtc_cfg = {
		.rtc_load = 1,
		.rtc_enable = 1,
		.etc_enable = 1,
	};

	rtc.seconds_47_34 = ts->tv_sec >> 34;
	rtc.seconds_33_0 = ts->tv_sec & (BIT(34) - 1);
	rtc.nanoseconds = ts->tv_nsec;

	cass_write(hw, C_MB_CFG_RTC_TIME, &rtc, C_MB_CFG_RTC_TIME_SIZE);
	cass_write(hw, C_MB_CFG_RTC, &rtc_cfg, sizeof(rtc_cfg));
}

static void read_current_time(struct cass_dev *hw,
			      union c_mb_sts_rtc_current_time *rtc)
{
	cass_read(hw, C_MB_STS_RTC_CURRENT_TIME, rtc, sizeof(*rtc));
}

/* Transform a time retrieved with read_current_time(). */
static void current_time_to_ts(const union c_mb_sts_rtc_current_time *rtc,
			       struct timespec64 *ts)
{
	ts->tv_sec = rtc->seconds_47_34;
	ts->tv_sec <<= 34;
	ts->tv_sec |= rtc->seconds_33_0;
	ts->tv_nsec = rtc->nanoseconds;
}

static void adjust_rtc(struct cass_dev *hw, s64 delta)
{
	const union c_mb_cfg_rtc rtc_cfg = {
		.rtc_update = 1,
		.rtc_enable = 1,
		.etc_enable = 1,
	};
	struct timespec64 rtc_ts;
	struct timespec64 delta_ts;

	if (abs(delta) > RTC_MAX_ADJ) {
		union c_mb_sts_rtc_current_time rtc;

		read_current_time(hw, &rtc);
		current_time_to_ts(&rtc, &rtc_ts);
		delta_ts = ns_to_timespec64(delta);
		rtc_ts = timespec64_add(rtc_ts, delta_ts);
		write_rtc(hw, &rtc_ts);
	} else {
		union c_mb_cfg_rtc_time rtc = {};

		if (delta > 0)
			rtc.nanoseconds = delta;
		else
			rtc.nanoseconds = BIT(RTC_NS_WIDTH) + delta;

		cass_write(hw, C_MB_CFG_RTC_TIME, &rtc,
			   C_MB_CFG_RTC_TIME_SIZE);
		cass_write(hw, C_MB_CFG_RTC, &rtc_cfg, sizeof(rtc_cfg));
	}
}

static void adjust_rtc_freq(struct cass_dev *hw, long scaled_ppm)
{
	union c_mb_cfg_rtc_inc rtc_inc = {};
	bool neg_adj;
	u64 frac_ns;
	u64 adj_ns;

	if (scaled_ppm < 0) {
		neg_adj = true;
		scaled_ppm = -scaled_ppm;
	} else {
		neg_adj = false;
	}

	/* From ptp_clock.c:
	 *    ppb = scaled_ppm * 10^3 * 2^-16
	 *
	 * adj_ns = ppb * 2^32 / clk_hz
	 *     = scaled_ppm * 10^3 * 2^-16 * 2^32 / clk_hz
	 *     = scaled_ppm * 65536000 / clk_hz
	 *     = scaled_ppm << 16 / clk_khz
	 */
	if (scaled_ppm) {
		adj_ns = scaled_ppm;
		adj_ns <<= 16;
		if (cass_version(hw, CASSINI_1))
			adj_ns = div_u64(adj_ns, C1_CLK_FREQ_HZ / 1000);
		else
			adj_ns = div_u64(adj_ns, C2_CLK_FREQ_HZ / 1000);
	} else {
		adj_ns = 0;
	}

	/* Set the default value for the clock frequency */
	if (cass_version(hw, CASSINI_1))
		frac_ns = RTC_MAX_FNS;	/* i.e. 1ns */
	else
		frac_ns = 3904515724; /* 2^32 * 1.0 Ghz / 1.1Ghz */

	/* And adjust it */
	if (neg_adj)
		frac_ns -= adj_ns;
	else
		frac_ns += adj_ns;

	if (frac_ns >= RTC_MAX_FNS) {
		rtc_inc.nanoseconds = 1;
		rtc_inc.fractional_ns = frac_ns - RTC_MAX_FNS;
	} else {
		rtc_inc.nanoseconds = 0;
		rtc_inc.fractional_ns = frac_ns;
	}

	cass_write(hw, C_MB_CFG_RTC_INC, &rtc_inc, sizeof(rtc_inc));

	/* Perform NOP RTC update so Cassini loads the new RTC_INC value. */
	adjust_rtc(hw, 0);
}

/* Initialize the time keeping CSRs before the clock is
 * started. Hopefully it's close enough to the RTC start that the
 * small time difference won't matter. RT_OFFSET could be calculated
 * later to compensate if needed.
 */
static void set_elapsed_time(struct cass_dev *hw, const struct timespec64 *ts)
{
	struct eltime {
		unsigned int elapsed_time;
		unsigned int time_offset_ns;
		unsigned int rt_offset;
	} time_init[] = {
		{ C_ATU_ERR_ELAPSED_TIME, 4, 0 },
		{ C_CQ_ERR_ELAPSED_TIME, 8, 0 },
		{ C_EE_ERR_ELAPSED_TIME, 7, C_EE_CFG_RT_OFFSET },
		{ C_HNI_ERR_ELAPSED_TIME, 12, C_HNI_CFG_RT_OFFSET },
		{ C_HNI_PML_ERR_ELAPSED_TIME, 13, 0 },
		{ C_IXE_ERR_ELAPSED_TIME, 17, 0 },
		{ C_LPE_ERR_ELAPSED_TIME, 11, 0 },
		{ C_MB_ERR_ELAPSED_TIME, 3, C_MB_CFG_RT_OFFSET },
		{ C_MST_ERR_ELAPSED_TIME, 13, C_MST_CFG_RT_OFFSET },
		{ C_OXE_ERR_ELAPSED_TIME, 18, C_OXE_CFG_RT_OFFSET },
		{ C_PARBS_ERR_ELAPSED_TIME, 9, C_PARBS_CFG_RT_OFFSET },
		{ C_PCT_ERR_ELAPSED_TIME, 10, 0 },
		{ C_PI_ERR_ELAPSED_TIME, 6, C_PI_CFG_RT_OFFSET },
		{ C_PI_IPD_ERR_ELAPSED_TIME, 15, C_PI_IPD_CFG_RT_OFFSET },
		{ C_RMU_ERR_ELAPSED_TIME, 15, 0 },
	};
	int i;

	for (i = 0; i < ARRAY_SIZE(time_init); i++) {
		struct eltime *t = &time_init[i];
		struct timespec64 offset_ts = *ts;
		union c_mb_err_elapsed_time err_elapsed_time;

		timespec64_add_ns(&offset_ts, time_init->time_offset_ns);
		err_elapsed_time.nanoseconds = offset_ts.tv_nsec,
		err_elapsed_time.seconds = offset_ts.tv_sec,

		cass_write(hw, t->elapsed_time, &err_elapsed_time,
			   sizeof(err_elapsed_time));
		if (t->rt_offset)
			cass_clear(hw, t->rt_offset, C_MB_CFG_RT_OFFSET_SIZE);
	}
}

static void init_rtc(struct cass_dev *hw)
{
	struct timespec64 ts = {};

	adjust_rtc_freq(hw, 0);

	ktime_get_real_ts64(&ts);
	set_elapsed_time(hw, &ts);
	write_rtc(hw, &ts);
}

static void fini_rtc(struct cass_dev *hw)
{
	cass_clear(hw, C_MB_CFG_RTC, C_MB_CFG_RTC_SIZE);
}

static int cass_ptp_adjfine(struct ptp_clock_info *ptp, long scaled_ppm)
{
	struct cass_dev *hw = container_of(ptp, struct cass_dev, ptp_info);
	unsigned long flags;

	spin_lock_irqsave(&hw->rtc_lock, flags);
	adjust_rtc_freq(hw, scaled_ppm);
	spin_unlock_irqrestore(&hw->rtc_lock, flags);

	return 0;
}

static int cass_ptp_adjtime(struct ptp_clock_info *ptp, s64 delta)
{
	struct cass_dev *hw = container_of(ptp, struct cass_dev, ptp_info);
	unsigned long flags;

	spin_lock_irqsave(&hw->rtc_lock, flags);
	adjust_rtc(hw, delta);
	spin_unlock_irqrestore(&hw->rtc_lock, flags);

	return 0;
}

static int cass_ptp_gettimex64(struct ptp_clock_info *ptp,
			       struct timespec64 *ts,
			       struct ptp_system_timestamp *sts)
{
	struct cass_dev *hw = container_of(ptp, struct cass_dev, ptp_info);
	union c_mb_sts_rtc_current_time rtc;
	unsigned long flags;

	spin_lock_irqsave(&hw->rtc_lock, flags);

	ptp_read_system_prets(sts);
	read_current_time(hw, &rtc);
	ptp_read_system_postts(sts);

	spin_unlock_irqrestore(&hw->rtc_lock, flags);

	current_time_to_ts(&rtc, ts);

	return 0;
}

static int cass_ptp_settime64(struct ptp_clock_info *ptp,
			      const struct timespec64 *ts)
{
	struct cass_dev *hw = container_of(ptp, struct cass_dev, ptp_info);
	unsigned long flags;

	spin_lock_irqsave(&hw->rtc_lock, flags);
	write_rtc(hw, ts);
	spin_unlock_irqrestore(&hw->rtc_lock, flags);

	return 0;
}

static int cass_ptp_enable(struct ptp_clock_info *ptp,
			   struct ptp_clock_request *request, int on)
{
	return -EOPNOTSUPP;
}

static int cass_ptp_verify(struct ptp_clock_info *ptp, unsigned int pin,
			   enum ptp_pin_function func, unsigned int chan)
{
	return -EOPNOTSUPP;
}

static struct ptp_clock_info cass_ptp_info = {
	.owner		= THIS_MODULE,
	.name		= "Cassini clock",
	.max_adj	= RTC_MAX_FNS - 1, /* Max. freq. adj. in  ppb */
	.adjfine	= cass_ptp_adjfine,
	.adjtime	= cass_ptp_adjtime,
	.gettimex64	= cass_ptp_gettimex64,
	.settime64	= cass_ptp_settime64,
	.enable		= cass_ptp_enable,
	.verify		= cass_ptp_verify,
};

int cass_ptp_init(struct cass_dev *hw)
{
	spin_lock_init(&hw->rtc_lock);
	hw->ptp_info = cass_ptp_info;

	init_rtc(hw);

	hw->ptp_clock = ptp_clock_register(&hw->ptp_info,
					   &hw->cdev.pdev->dev);
	if (IS_ERR(hw->ptp_clock)) {
		fini_rtc(hw);
		return PTR_ERR(hw->ptp_clock);
	}

	return 0;
}

void cass_ptp_fini(struct cass_dev *hw)
{
	ptp_clock_unregister(hw->ptp_clock);
	fini_rtc(hw);
}
