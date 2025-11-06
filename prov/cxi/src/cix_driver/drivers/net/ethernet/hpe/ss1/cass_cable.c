// SPDX-License-Identifier: GPL-2.0
/* Copyright 2021-2022,2025 Hewlett Packard Enterprise Development LP */

/* Cassini HSN cable support */

#include <linux/sbl.h>
#include "cass_core.h"
#include "cass_cable.h"

static int cass_hdsh_get_format(struct cass_dev *hw, u8 *qsfp_format);
static void cass_hdsh_get_media(struct cass_dev *hw,
				struct sbl_media_attr *attr, u8 qsfp_format);
static void cass_hdsh_get_info(struct cass_dev *hw,
			       struct sbl_media_attr *attr, u8 qsfp_format);
static void cass_hdsh_get_length(struct cass_dev *hw,
				 struct sbl_media_attr *attr, u8 qsfp_format);
static void cass_hdsh_get_vendor(struct cass_dev *hw,
				 struct sbl_media_attr *attr, u8 qsfp_format);
static int cass_link_headshell_down_no_media(struct cass_dev *hw, int reason);

/**
 * cass_is_cable_present() - Check if we have an HSN cable installed
 *
 * @hw: the device
 *
 * Return: 1 if cable present, else 0
 */
int cass_is_cable_present(struct cass_dev *hw)
{
	if (!HW_PLATFORM_ASIC(hw))
		return 1;
	return (hw->qsfp_eeprom_page_len != 0);
}

/**
 * cass_parse_heashell_data() - Read the QSFP EEPROM and parse into attr
 *
 * @hw: the device
 * @attr: pointer to struct to initialize with cable information
 * @qsfp_format: What spec this cable CSR space is organized according to.
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_parse_heashell_data(struct cass_dev *hw, struct sbl_media_attr *attr,
			     u8 *qsfp_format)
{
	int rc = 0;

	mutex_lock(&hw->qsfp_eeprom_lock);

	if (hw->qsfp_eeprom_page_len <= 0) {
		cxidev_err(&hw->cdev,
			   "Failed to parse headshell data. EEPROM data unavailable!\n");
		rc = -EIO;
		goto out_unlock;
	}

	/* Start by determining the cable format */
	rc = cass_hdsh_get_format(hw, qsfp_format);
	if (rc)
		goto out_unlock;

	/* Read out media type information and parse into attr */
	cass_hdsh_get_media(hw, attr, *qsfp_format);

	/* Read out the supported speed modes into attr */
	cass_hdsh_get_info(hw, attr, *qsfp_format);

	/* Read out media length information into attr */
	cass_hdsh_get_length(hw, attr, *qsfp_format);

	/* Read out vendor information and parse into attr */
	cass_hdsh_get_vendor(hw, attr, *qsfp_format);

 out_unlock:
	mutex_unlock(&hw->qsfp_eeprom_lock);
	return rc;
}

/**
 * cass_hdsh_get_format() - Read the QSFP EEPROM and parse out the media format
 *
 * @hw: the device
 * @qsfp_format: pointer to spec this cable CSR space is organized according to
 *
 * Return: 0 on success, negative errno on failure
 */
static int cass_hdsh_get_format(struct cass_dev *hw, u8 *qsfp_format)
{
	if (hw->qsfp_eeprom_page0[QSFP_IDENTIFIER_OFFSET] >= SFF8024_TYPE_QSFPDD) {
		// What revision compliance do they claim
		if (hw->qsfp_eeprom_page0[QSFP_REV_CMPL_OFFSET] < 0x27)
			/* Very similar to CMIS 2.0, might need to distinguish
			 *  someday?
			 */
			*qsfp_format = QDD_SFF8636;
		else {
			*qsfp_format = QDD_CMIS;
			/* CMIS V3.0-V4.0 are mostly compatible.
			 * Version-specific differences are handled in the code.
			 */
			/* allow CMIS v5.1 */
			if (hw->qsfp_eeprom_page0[QSFP_REV_CMPL_OFFSET] > 0x51) {
				cxidev_err(&hw->cdev,
					   "Failed to parse headshell data. Bad CMIS rev (%d)!\n",
					   hw->qsfp_eeprom_page0[QSFP_REV_CMPL_OFFSET]);
				return -EMEDIUMTYPE;
			}
		}
	} else
		// Treat anything else as SFF-8636 Format
		*qsfp_format = QDD_SFF8636;

	return 0;
}

/**
 * cass_hdsh_get_media() - Read the QSFP EEPROM and parse out the media type
 *
 * @hw: the device
 * @attr: pointer to struct to initialize with cable information
 * @qsfp_format: What spec this cable CSR space is organized according to.
 *
 * Return: 0 on success, negative errno on failure
 */
static void cass_hdsh_get_media(struct cass_dev *hw,
				struct sbl_media_attr *attr, u8 qsfp_format)
{
	u8 qsfp_media;

	if (qsfp_format == QDD_CMIS)
		qsfp_media = hw->qsfp_eeprom_page0[CMIS_MEDIA_TYPE_OFFSET];
	else /* QDD_SFF8636 */
		qsfp_media = hw->qsfp_eeprom_page0[SFF8436_MEDIA_TYPE_OFFSET];
	if (qsfp_media >= MEDIA_COPPER_UNEQ)
		attr->media = SBL_LINK_MEDIA_ELECTRICAL;
	else
		attr->media = SBL_LINK_MEDIA_OPTICAL;
}

/**
 * cass_hdsh_get_info() - Read the QSFP EEPROM and parse out the media info
 *
 * @hw: the device
 * @attr: pointer to struct to initialize with cable information
 * @qsfp_format: What spec this cable CSR space is organized according to.
 *
 * Return: 0 on success, negative errno on failure
 */
static void cass_hdsh_get_info(struct cass_dev *hw,
			       struct sbl_media_attr *attr, u8 qsfp_format)
{
	if (attr->media == SBL_LINK_MEDIA_ELECTRICAL)
		// For copper cables, we support all link modes
		attr->info = SBL_MEDIA_INFO_SUPPORTS_BS_200G |
			SBL_MEDIA_INFO_SUPPORTS_BJ_100G |
			SBL_MEDIA_INFO_SUPPORTS_CD_100G |
			SBL_MEDIA_INFO_SUPPORTS_CD_50G;
	else {
		if (qsfp_format == QDD_CMIS)
			/* TODO - CMIS support is not immediately required */
			return;
		/* else: QDD_SFF8636 */
		/* SFF8636 is not supported, so advertise all link mode
		 *  are supported.
		 */
		attr->info = SBL_MEDIA_INFO_SUPPORTS_BS_200G |
			SBL_MEDIA_INFO_SUPPORTS_BJ_100G |
			SBL_MEDIA_INFO_SUPPORTS_CD_100G |
			SBL_MEDIA_INFO_SUPPORTS_CD_50G;
	}
}


/**
 * cass_hdsh_get_length() - Read the QSFP EEPROM and parse out the media length
 *
 * @hw: the device
 * @attr: pointer to struct to initialize with cable information
 * @qsfp_format: What spec this cable CSR space is organized according to.
 *
 * Return: 0 on success, negative errno on failure
 */
static void cass_hdsh_get_length(struct cass_dev *hw,
				 struct sbl_media_attr *attr, u8 qsfp_format)
{
	u8 qsfp_length;
	int mult, qsfp_length_mm;

	if (qsfp_format == QDD_CMIS) {
		qsfp_length = hw->qsfp_eeprom_page0[CMIS_CABLE_LEN_OFFSET];
		mult = (qsfp_length & 0xC0) >> 6;
		qsfp_length &= 0x3F;
		switch (mult) {
		case 0:
			attr->len = qsfp_length * 10;
			break;
		case 1:
			attr->len = qsfp_length * 100;
			break;
		case 2:
			attr->len = qsfp_length * 1000;
			break;
		case 3:
			attr->len = qsfp_length * 10000;
			break;
		default:
			attr->len = 0;
			break;
		}
	} else { /* QDD_SFF8636 */
		qsfp_length = hw->qsfp_eeprom_page0[SFF8436_LINK_LEN_OFFSET];
		attr->len = qsfp_length * 100;
	}

	/* TODO - sbl_link_len will be converted to len in cm, rather than
	 *  this enum. Remove this section when that happens
	 */
	qsfp_length_mm = attr->len * 10;
	switch (qsfp_length_mm) {
	case 300:
		attr->len = SBL_LINK_LEN_000_300; break;
	case 400:
		attr->len = SBL_LINK_LEN_000_400; break;
	case 750:
		attr->len = SBL_LINK_LEN_000_750; break;
	case 800:
		attr->len = SBL_LINK_LEN_000_800; break;
	case 1000:
		attr->len = SBL_LINK_LEN_001_000; break;
	case 1100:
		attr->len = SBL_LINK_LEN_001_100; break;
	case 1150:
		attr->len = SBL_LINK_LEN_001_150; break;
	case 1200:
		attr->len = SBL_LINK_LEN_001_200; break;
	case 1400:
		attr->len = SBL_LINK_LEN_001_400; break;
	case 1420:
		attr->len = SBL_LINK_LEN_001_420; break;
	case 1500:
		attr->len = SBL_LINK_LEN_001_500; break;
	case 1600:
		attr->len = SBL_LINK_LEN_001_600; break;
	case 1640:
		attr->len = SBL_LINK_LEN_001_640; break;
	case 1700:
		attr->len = SBL_LINK_LEN_001_700; break;
	case 1900:
		attr->len = SBL_LINK_LEN_001_900; break;
	case 1910:
		attr->len = SBL_LINK_LEN_001_910; break;
	case 2000:
		attr->len = SBL_LINK_LEN_002_000; break;
	case 2100:
		attr->len = SBL_LINK_LEN_002_100; break;
	case 2130:
		attr->len = SBL_LINK_LEN_002_130; break;
	case 2200:
		attr->len = SBL_LINK_LEN_002_200; break;
	case 2300:
		attr->len = SBL_LINK_LEN_002_300; break;
	case 2390:
		attr->len = SBL_LINK_LEN_002_390; break;
	case 2400:
		attr->len = SBL_LINK_LEN_002_400; break;
	case 2500:
		attr->len = SBL_LINK_LEN_002_500; break;
	case 2600:
		attr->len = SBL_LINK_LEN_002_600; break;
	case 2620:
		attr->len = SBL_LINK_LEN_002_620; break;
	case 2700:
		attr->len = SBL_LINK_LEN_002_700; break;
	case 2800:
		attr->len = SBL_LINK_LEN_002_800; break;
	case 2900:
		attr->len = SBL_LINK_LEN_002_900; break;
	case 2990:
		attr->len = SBL_LINK_LEN_002_990; break;
	case 3000:
		attr->len = SBL_LINK_LEN_003_000; break;
	case 4000:
		attr->len = SBL_LINK_LEN_004_000; break;
	case 5000:
		attr->len = SBL_LINK_LEN_005_000; break;
	case 6000:
		attr->len = SBL_LINK_LEN_006_000; break;
	case 7000:
		attr->len = SBL_LINK_LEN_007_000; break;
	case 8000:
		attr->len = SBL_LINK_LEN_008_000; break;
	case 10000:
		attr->len = SBL_LINK_LEN_010_000; break;
	case 14000:
		attr->len = SBL_LINK_LEN_014_000; break;
	case 15000:
		attr->len = SBL_LINK_LEN_015_000; break;
	case 19000:
		attr->len = SBL_LINK_LEN_019_000; break;
	case 25000:
		attr->len = SBL_LINK_LEN_025_000; break;
	case 30000:
		attr->len = SBL_LINK_LEN_030_000; break;
	case 35000:
		attr->len = SBL_LINK_LEN_035_000; break;
	case 50000:
		attr->len = SBL_LINK_LEN_050_000; break;
	case 75000:
		attr->len = SBL_LINK_LEN_075_000; break;
	case 100000:
		attr->len = SBL_LINK_LEN_100_000; break;
	default:
		attr->len = SBL_LINK_LEN_INVALID;
		cxidev_warn(&hw->cdev, "No valid enum found for link length %d!",
			    qsfp_length_mm);
		break;
	}
}

/**
 * cass_hdsh_get_vendor() - Read the QSFP EEPROM and parse out the media vendor
 *
 * @hw: the device
 * @attr: pointer to struct to initialize with cable information
 * @qsfp_format: What spec this cable CSR space is organized according to.
 *
 * Return: 0 on success, negative errno on failure
 */
static void cass_hdsh_get_vendor(struct cass_dev *hw,
				 struct sbl_media_attr *attr, u8 qsfp_format)
{
	const char *qsfp_vendor;

	if (qsfp_format == QDD_CMIS)
		qsfp_vendor = &(hw->qsfp_eeprom_page0[CMIS_VENDOR_OFFSET]);
	else /* QDD_SFF8636 */
		qsfp_vendor = &(hw->qsfp_eeprom_page0[SFF8436_VENDOR_OFFSET]);

	if (strnstr(qsfp_vendor, QDD_VENDOR_TE, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_TE;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_LEONI, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_LEONI;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_MOLEX, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_MOLEX;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_HISENSE, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_HISENSE;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_DUST_PHOTONICS, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_DUST_PHOTONICS;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_FINISAR, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_FINISAR;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_LUXSHARE, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_LUXSHARE;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_FIT, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_FIT;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_FT, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_FT;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_MELLANOX, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_MELLANOX;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_HITACHI, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_HITACHI;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_HPE, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_HPE;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_CLOUD_LIGHT, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SBL_LINK_VENDOR_CLOUD_LIGHT;
	else if (strnstr(qsfp_vendor, QDD_VENDOR_AMPHENOL, QSFP_VENDOR_MAX_STR_LEN) != NULL)
		attr->vendor = SL_MEDIA_VENDOR_AMPHENOL;
	else {
		cxidev_warn(&hw->cdev, "No valid enum found for link Vendor %16s",
			    qsfp_vendor);
		attr->vendor = SBL_LINK_VENDOR_INVALID;
	}
}

/**
 * cass_headshell_power_up() - Power headshell up to high power mode
 *
 * @hw: the device
 * @qsfp_format: What spec this cable CSR space is organized according to.
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_headshell_power_up(struct cass_dev *hw, u8 qsfp_format)
{
	u8 qsfp_extid, data;
	int rc = 0;

	mutex_lock(&hw->qsfp_eeprom_lock);

	// Full power up of all data paths + high power mode
	if (qsfp_format == QDD_CMIS) {
		/* TODO: aapl script will do the power up */
		rc = 0;
	} else if (qsfp_format == QDD_SFF8636) {
		// If not, power up the QSFP
		data = (SFF8636_PWR_CLASS_567_ENABLE | SFF8636_PWR_OVERRIDE);
		qsfp_extid = hw->qsfp_eeprom_page0[SFF8436_EXT_IDENTIFIER_OFFSET];
		// If the module supports Class8+ set that enable bit too
		if (qsfp_extid & SFF8636_PWR_CLASS_8)
			data |= SFF8636_PWR_CLASS_8_ENABLE;
		rc = uc_cmd_qsfp_write(hw, 0, SFF8436_POWER_DATA_OFFSET,
				       &data, sizeof(data));
	} else
		rc = -EINVAL;

	mutex_unlock(&hw->qsfp_eeprom_lock);
	return rc;
}

/**
 * cass_link_headshell_insert() - Headshell insert
 *
 * @hw: the device
 * @attr: pointer to struct containing cable information
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_link_headshell_insert(struct cass_dev *hw,
			       struct sbl_media_attr *attr)
{
	int err;

	cass_phy_set_state(CASS_PHY_NOLINK, hw);
	cxidev_dbg(&hw->cdev, "headshell insert: phy state = %d\n", hw->phy.state);

	if (hw->port->hstate == CASS_HEADSHELL_STATUS_PRESENT) {
		cxidev_err(&hw->cdev, "headshell insert - already present\n");
		return -EUCLEAN;
	}

	if (hw->port->config_state & CASS_MEDIA_CONFIGURED) {
		cxidev_err(&hw->cdev,
			   "headshell insert - media already configured\n");
		return -EUCLEAN;
	}

	err = hw->link_ops->media_config(hw, attr);
	if (err) {
		cxidev_err(&hw->cdev, "headshell insert - media config failed [%d]\n",
			   err);
		hw->port->config_state &= ~CASS_MEDIA_CONFIGURED;
		return err;
	}

	/*
	 * check any existing link config is compatible with the new media type
	 *
	 *    if not unconfigure link so it must be reconfigured again
	 *
	 */
	if (hw->port->config_state & CASS_LINK_CONFIGURED) {
		switch (hw->port->lattr.bl.config_target) {
		case SBL_BASE_LINK_CONFIG_PEC:
			if (attr->media != SBL_LINK_MEDIA_ELECTRICAL) {
				cxidev_warn(&hw->cdev,
					    "headshell insert - link attr PEC incompatible with new media");
				hw->port->config_state &= ~CASS_LINK_CONFIGURED;
			}
			break;
		case SBL_BASE_LINK_CONFIG_AOC:
			if (attr->media != SBL_LINK_MEDIA_OPTICAL) {
				cxidev_warn(&hw->cdev,
					    "headshell insert - link attr AOC incompatible with new media");
				hw->port->config_state &= ~CASS_LINK_CONFIGURED;
			}
			break;
		default:
			cxidev_warn(&hw->cdev,
				    "headshell insert - link attr incompatible with new media");
			hw->port->config_state &= ~CASS_LINK_CONFIGURED;
			break;
		}
	}

	hw->port->hstate = CASS_HEADSHELL_STATUS_PRESENT;
	hw->port->config_state |= CASS_MEDIA_CONFIGURED;

	/*
	 * if the link is supposed to be up try to restart it
	 */
	if ((cass_link_get_state(hw) == CASS_LINK_STATUS_DOWN) &&
	    (hw->port->lmon_dirn == CASS_LMON_DIRECTION_UP)) {

		if ((hw->port->lattr.options &
		     CASS_LINK_OPT_RESTART_ON_INSERT)) {
			cxidev_dbg(&hw->cdev, "starting link after insert\n");
			cass_lmon_request_up(hw);
		}
	}

	return 0;
}

/**
 * cass_link_headshell_remove() - Headshell remove
 *
 *    Headshell must already be present and media configured.
 *    Stop link if running and unconfigure media
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_link_headshell_remove(struct cass_dev *hw)
{
	int err;

	cass_phy_set_state(CASS_PHY_HEADSHELL_REMOVED, hw);
	cxidev_dbg(&hw->cdev, "headshell remove: phy state = %d\n", hw->phy.state);

	switch (hw->port->hstate) {
	case CASS_HEADSHELL_STATUS_NOT_PRESENT:
		cxidev_err(&hw->cdev, "headshell remove - not present\n");
		return -EUCLEAN;

	case CASS_HEADSHELL_STATUS_PRESENT:
		if (!(hw->port->config_state & CASS_MEDIA_CONFIGURED)) {
			cxidev_err(&hw->cdev,
				   "headshell remove - media not configured\n");
			return -EUCLEAN;
		}
		break;

	case CASS_HEADSHELL_STATUS_ERROR:
		break;

	default:
		cxidev_err(&hw->cdev, "headshell remove - unrecognised state\n");
		return -EUCLEAN;
	}

	/* stop link coming up */
	cass_lmon_set_up_pause(hw, true);

	/* setup new state */
	hw->port->hstate = CASS_HEADSHELL_STATUS_NOT_PRESENT;
	cass_link_set_led(hw);

	/* ensure link down and no media configured */
	err = cass_link_headshell_down_no_media(hw,
					CASS_DOWN_ORIGIN_HEADSHELL_REMOVED);
	if (err) {
		cxidev_err(&hw->cdev,
			   "headshell remove - down_no_media failed [%d]\n",
			   err);
		cass_lmon_set_up_pause(hw, false);
		return err;
	}

	cass_lmon_set_up_pause(hw, false);
	return 0;
}

/**
 * cass_link_headshell_error() - Headshell error
 *
 *    Any start state
 *    Stop link and unconfigure media
 *
 * @hw: the device
 *
 * Return: 0 on success, negative errno on failure
 */
int cass_link_headshell_error(struct cass_dev *hw)
{
	int err;

	cass_phy_set_state(CASS_PHY_HEADSHELL_REMOVED, hw);
	cxidev_dbg(&hw->cdev, "headshell error: phy state = %d\n", hw->phy.state);

	/* stop link coming up */
	cass_lmon_set_up_pause(hw, true);

	/* ensure link down and no media configured */
	err = cass_link_headshell_down_no_media(hw,
					CASS_DOWN_ORIGIN_HEADSHELL_ERROR);
	if (err) {
		cxidev_err(&hw->cdev,
			   "headshell error - down_no_media failed [%d]\n",
			   err);
		cass_lmon_set_up_pause(hw, false);
		return err;
	}

	/* setup new state */
	hw->port->hstate = CASS_HEADSHELL_STATUS_ERROR;
	cass_link_set_led(hw);

	cass_lmon_set_up_pause(hw, false);
	return 0;
}

/**
 * cass_link_headshell_down_no_media() - take link down and unconfigure cable
 *
 * Common headshell remove/error code to stop link and unconfigure media
 *
 * @hw: the device
 * @reason: the reason the link is going down
 *
 * Return: 0 on success, negative errno on failure
 */
static int cass_link_headshell_down_no_media(struct cass_dev *hw, int reason)
{
	int err;

	cxidev_dbg(&hw->cdev, "headshell down no media\n");

	/* ensure link down  */
	spin_lock(&hw->port->lock);
	switch (hw->port->lstate) {
	case CASS_LINK_STATUS_DOWN:
	case CASS_LINK_STATUS_ERROR:
	case CASS_LINK_STATUS_UNCONFIGURED:
	case CASS_LINK_STATUS_UNKNOWN:
		cxidev_dbg(&hw->cdev, "headshell down - already link down (%s)\n",
			   cass_link_state_str(hw->port->lstate));
		spin_unlock(&hw->port->lock);
		break;
	default:
		spin_unlock(&hw->port->lock);
		err = cass_link_async_down_wait(hw, reason);
		if (err) {
			cxidev_err(&hw->cdev,
				   "headshell down - down_wait failed [%d]\n",
				   err);
			return err;
		}
		break;
	}

	/* remove any media configuration */
	if (hw->port->config_state & CASS_MEDIA_CONFIGURED) {
		err = hw->link_ops->media_unconfig(hw);
		if (err) {
			cxidev_err(&hw->cdev, "headshell down - media unconfig failed [%d]\n", err);
			return err;
		}
		hw->port->config_state &= ~CASS_MEDIA_CONFIGURED;
	}

	return 0;
}

/*
 * Direct media configuration
 *
 *   Called by hms to set media for backplane media
 *
 *   If you are playing with the backplane (i.e. changing compute blades)
 *   then leave it to the user to restart the link
 */
int cass_link_media_config(struct cass_dev *hw, struct sbl_media_attr *attr)
{
	int err;

	cxidev_dbg(&hw->cdev, "media config\n");

	if (hw->port->hstate != CASS_HEADSHELL_STATUS_NOT_PRESENT) {
		cxidev_dbg(&hw->cdev, "media config - bad state %s\n",
			   cass_link_headshell_state_str(hw->port->hstate));
		return -EUCLEAN;
	}

	if (hw->port->config_state & CASS_MEDIA_CONFIGURED) {
		cxidev_dbg(&hw->cdev, "media config - media already configured\n");
		return -EUCLEAN;
	}

	err = hw->link_ops->media_config(hw, attr);
	if (err) {
		cxidev_dbg(&hw->cdev, "media config - media config failed [%d]\n",
			   err);
		hw->port->config_state &= ~CASS_MEDIA_CONFIGURED;
		return err;
	}

	hw->port->hstate = CASS_HEADSHELL_STATUS_NO_DEVICE;
	hw->port->config_state |= CASS_MEDIA_CONFIGURED;

	return 0;
}

/**
 * cass_link_headshell_state_str() - Get a string for the headshell state
 *
 * Return a text string describing a headshell status
 *
 * @state: input enum of target headshell state
 *
 * Return: This function will always return a text string
 */
const char *cass_link_headshell_state_str(enum cass_headshell_status state)
{
	switch (state) {
	case CASS_HEADSHELL_STATUS_UNKNOWN:     return "unknown";
	case CASS_HEADSHELL_STATUS_PRESENT:     return "present";
	case CASS_HEADSHELL_STATUS_NOT_PRESENT: return "not present";
	case CASS_HEADSHELL_STATUS_NO_DEVICE:   return "no device";
	default:                                return "unrecognised";
	}
}
