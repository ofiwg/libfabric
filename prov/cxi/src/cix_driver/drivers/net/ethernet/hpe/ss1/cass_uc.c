// SPDX-License-Identifier: GPL-2.0
/* Copyright 2019-2021 Hewlett Packard Enterprise Development LP */

/* Communication with embedded µC
 *
 * The driver prepares a message with a command and signals the
 * µC. When the µC has processed the command, it raises
 * C_PI_ERR_FLG.UC_ATTENTION[0]. The callback the driver has
 * registered for these bits signals a completion, and the response
 * can be processed.
 *
 * C_PI_ERR_FLG.UC_ATTENTION[1] is used for asynchronous events, like
 * cable status change.
 *
 * The uc_mbox_mutex protects the uc_req and uc_resp buffers, as well
 * as the communication.
 */

#include <linux/workqueue.h>
#include <linux/firmware.h>
#include <linux/delay.h>
#include <linux/etherdevice.h>

#include "cass_core.h"
#include "cass_cable.h"
#include "cass_ss1_debugfs.h"

/* Reserved DMAC registers for communication.
 * Range is [FIRST, FIRST+16)
 */
#define HOST_TO_UC_FIRST 992
#define UC_TO_HOST_FIRST 1008

/* Common command initialization */
void uc_prepare_comm(struct cass_dev *hw)
{
	/* Clear the uC to host buffer (C_PI_CFG_DMAC_DESC[1008:1023]) */
	memset(&hw->uc_req, 0, sizeof(hw->uc_req));
	memset(&hw->uc_resp, 0, sizeof(hw->uc_resp));
	cass_write(hw, C_PI_CFG_DMAC_DESC(UC_TO_HOST_FIRST),
		   &hw->uc_resp, sizeof(hw->uc_resp));
	reinit_completion(&hw->uc_attention0_comp);
}

/* Signal a new command is ready, and wait for the µC to raise
 * UC_ATTENTION[0] of C_PI_ERR_FLAG.
 *
 * Return 0 if the response is valid, or a negative errno on error.
 */
int uc_wait_for_response(struct cass_dev *hw)
{
	static const union c_hni_uc_sbr_intr srb_intr = {
		.send_sbr_dis = 1,
		.intr = 1,
	};
	int rc;

	hw->uc_req.type = CUC_TYPE_REQ;

	/* Write the request and ring the doorbell */
	cass_write(hw, C_PI_CFG_DMAC_DESC(HOST_TO_UC_FIRST),
		   &hw->uc_req, sizeof(hw->uc_req));
	cass_write(hw, C_HNI_UC_SBR_INTR, &srb_intr, sizeof(srb_intr));
	cass_flush_pci(hw);

	rc = wait_for_completion_interruptible_timeout(&hw->uc_attention0_comp,
						       20 * HZ);
	if (rc == 0)
		return -ETIMEDOUT;
	else if (rc < 0)
		return rc;

	/* Read the response */
	cass_read(hw, C_PI_CFG_DMAC_DESC(UC_TO_HOST_FIRST),
		  &hw->uc_resp, sizeof(hw->uc_resp));

	switch (hw->uc_resp.type) {
	case CUC_TYPE_RSP_SUCCESS:
		rc = 0;
		break;
	case CUC_TYPE_RSP_ERROR: {
		struct cuc_error_rsp_data *rsp_data =
			(struct cuc_error_rsp_data *)hw->uc_resp.data;
		rc = -rsp_data->error;
		if (rc == 0) {
			/* Be paranoid for now. Should not happen. */
			cxidev_warn_once(&hw->cdev, "Got error code 0 from uC\n");
			rc = -EIO;
		}
		break;
	}
	default:
		/* Be paranoid for now. Should not happen. */
		cxidev_warn_once(&hw->cdev, "Unexpected uC response type %u\n",
				 hw->uc_resp.type);
		rc = -EIO;
	}

	return rc;
}

/* Get the firmware version for the given target */
static void uc_cmd_get_fw_version(struct cass_dev *hw, unsigned int fw_target)
{
	struct device *dev = &hw->cdev.pdev->dev;
	struct cuc_get_firmware_version_req *req_data =
		(struct cuc_get_firmware_version_req *)hw->uc_req.data;
	int rc;

	if (!hw->uc_present)
		return;

	if (fw_target >= ARRAY_SIZE(hw->fw_versions))
		return;

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_FIRMWARE_VERSION;
	hw->uc_req.count = 1 + sizeof(*req_data);
	req_data->nic = CUC_MAC_THIS_NIC;
	req_data->fw_target = fw_target;
	req_data->from_flash = 0;
	req_data->slot = FW_SLOT_ACTIVE;

	rc = uc_wait_for_response(hw);
	if (rc) {
		cxidev_err(&hw->cdev,
			   "uC failed to return FW info for target %d: %d\n",
			   fw_target, rc);
		goto done;
	}

	if (hw->fw_versions[fw_target]) {
		devm_kfree(dev, hw->fw_versions[fw_target]);
		hw->fw_versions[fw_target] = NULL;
	}

	hw->fw_versions[fw_target] =
		devm_kasprintf(dev, GFP_KERNEL, "%.*s",
			       hw->uc_resp.count, hw->uc_resp.data);

	/* Target 2 is the main firmware version. */
	if (fw_target == 2 && hw->fw_versions[fw_target])
		cxidev_info(&hw->cdev, "uC firmware version %s\n",
			    hw->fw_versions[fw_target]);

done:
	mutex_unlock(&hw->uc_mbox_mutex);
}

/* Get the various uC firmware versions */
static void uc_cmd_get_fw_versions(struct cass_dev *hw)
{
	uc_cmd_get_fw_version(hw, FW_UC_APPLICATION);
	uc_cmd_get_fw_version(hw, FW_UC_BOOTLOADER);
	uc_cmd_get_fw_version(hw, FW_QSPI_BLOB);
	if (cass_version(hw, CASSINI_1))
		uc_cmd_get_fw_version(hw, FW_OPROM);
	uc_cmd_get_fw_version(hw, FW_CSR1);
	uc_cmd_get_fw_version(hw, FW_CSR2);
	uc_cmd_get_fw_version(hw, FW_SRDS);
}

/* Get the NIC ID */
int uc_cmd_get_nic_id(struct cass_dev *hw)
{
	const struct cuc_get_nic_id_rsp *rsp_data =
		(struct cuc_get_nic_id_rsp *)hw->uc_resp.data;
	int rc;

	if (!hw->uc_present)
		return -ENOTSUPP;

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_GET_NIC_ID;
	hw->uc_req.count = 1;

	rc = uc_wait_for_response(hw);
	if (!rc)
		hw->uc_nic = rsp_data->nic;
	else
		cxidev_err(&hw->cdev, "uC failed to return the NIC ID: %d\n", rc);

	mutex_unlock(&hw->uc_mbox_mutex);

	return rc;
}

/* Get the MAC address */
int uc_cmd_get_mac(struct cass_dev *hw)
{
	struct cuc_mac_req_data *req_data =
		(struct cuc_mac_req_data *)hw->uc_req.data;
	const struct cuc_mac_rsp_data *rsp_data =
		(struct cuc_mac_rsp_data *)hw->uc_resp.data;
	int rc;

	if (!hw->uc_present)
		return -ENOTSUPP;

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_GET_MAC;
	hw->uc_req.count = sizeof(struct cuc_mac_req_data) + 1;
	req_data->nic = CUC_MAC_THIS_NIC;

	rc = uc_wait_for_response(hw);
	if (!rc) {
		ether_addr_copy(hw->default_mac_addr, rsp_data->nic_mac);
	} else {
		if (rc == -ENOENT)
			cxidev_err(&hw->cdev,
				   "MAC address not yet programmed in uC\n");
		else
			cxidev_err(&hw->cdev,
				   "uC failed to return the MAC info: %d\n",
				   rc);
	}

	mutex_unlock(&hw->uc_mbox_mutex);

	return rc;
}

static void decode_fru_field_info(struct cass_dev *hw,
				  const struct pldm_fru_field *fru_field)
{
	struct device *dev = &hw->cdev.pdev->dev;

	if (fru_field->field_type == 0 ||
	    fru_field->field_type > PLDM_FRU_FIELD_VENDOR_IANA) {
		cxidev_info(&hw->cdev, "Unsupported FRU field type %u\n",
			    fru_field->field_type);
		return;
	}

	/* All supported fields are strings except two */
	if (fru_field->field_type == PLDM_FRU_FIELD_MANUFACTURE_DATE) {
		hw->fru_info[fru_field->field_type] = devm_kasprintf(
			dev, GFP_KERNEL, "%04u:%02u:%02u:%02u:%02u:%02u",
			*(u16 *)&fru_field->value[10], fru_field->value[9],
			fru_field->value[8], fru_field->value[7],
			fru_field->value[6], fru_field->value[5]);
	} else if (fru_field->field_type == PLDM_FRU_FIELD_VENDOR_IANA) {
		hw->fru_info[fru_field->field_type] = devm_kasprintf(
			dev, GFP_KERNEL, "%x", *(u32 *)fru_field->value);
	} else {
		hw->fru_info[fru_field->field_type] = devm_kasprintf(
			dev, GFP_KERNEL, "%.*s", fru_field->length,
			fru_field->value);
	}
}

static void decode_fru_info(struct cass_dev *hw)
{
	const struct pldm_fru_record *fru;
	const struct pldm_fru_field *fru_field;
	int len = hw->uc_resp.count - 1;
	unsigned int fru_num;

	if (len > CUC_DATA_BYTES) {
		cxidev_err(&hw->cdev, "uC returned length too big\n");
		return;
	}

	len -= sizeof(struct pldm_fru_record);
	if (len < 0) {
		cxidev_err(&hw->cdev, "uC returned invalid FRU length\n");
		return;
	}

	fru = (struct pldm_fru_record *)hw->uc_resp.data;

	if (fru->record_type != PLDM_FRU_RECORD_GENERAL) {
		cxidev_err(&hw->cdev, "uC returned unknown record type %x\n",
			   fru->record_type);
		return;
	}

	fru_field = (const struct pldm_fru_field *)fru->field_data;
	for (fru_num = 0; fru_num < fru->num_fields; fru_num++) {
		len -= sizeof(struct pldm_fru_field);
		if (len < 0) {
			cxidev_err(&hw->cdev,
				   "uC returned invalid FRU field header length\n");
			return;
		}
		len -= fru_field->length;
		if (len < 0) {
			cxidev_err(&hw->cdev,
				   "uC returned invalid FRU field length\n");
			return;
		}

		decode_fru_field_info(hw, fru_field);

		fru_field = (const struct pldm_fru_field *)
			(((uintptr_t)(fru_field + 1)) + fru_field->length);
	}
}

/* Get the FRU information */
int uc_cmd_get_fru(struct cass_dev *hw)
{
	int rc;

	if (HW_PLATFORM_Z1(hw)) {
		/* Add needed info for libcxi to run. */
		hw->fru_info[PLDM_FRU_FIELD_DESCRIPTION] =
			devm_kasprintf(&hw->cdev.pdev->dev, GFP_KERNEL, "Z1");
		return 0;
	}

	if (!hw->uc_present)
		return -ENOTSUPP;

	if (hw->fru_info_valid)
		return 0;

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_GET_FRU;
	hw->uc_req.count = 1;

	rc = uc_wait_for_response(hw);
	if (!rc) {
		hw->fru_info_valid = true;
		decode_fru_info(hw);
	} else {
		cxidev_err(&hw->cdev,
			   "uC failed to return the FRU info: %d\n", rc);
	}

	mutex_unlock(&hw->uc_mbox_mutex);
	return rc;
}

/* Read QSFP eeprom data for a given page */
int cxi_get_qsfp_data(struct cxi_dev *cdev, u32 offset, u32 len, u32 page, u8 *data)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cuc_qsfp_read_req_data *req_data =
		(struct cuc_qsfp_read_req_data *)hw->uc_req.data;
	const struct cuc_qsfp_read_rsp_data *rsp_data =
		(struct cuc_qsfp_read_rsp_data *)hw->uc_resp.data;
	int ret_len = len;
	int rc;

	if (!hw->uc_present ||
	    (hw->uc_platform != CUC_BOARD_TYPE_BRAZOS &&
	     hw->uc_platform != CUC_BOARD_TYPE_KENNEBEC &&
	     hw->uc_platform != CUC_BOARD_TYPE_SOUHEGAN &&
	     !HW_PLATFORM_NETSIM(hw)))
		return -ENOMEDIUM;

	mutex_lock(&hw->uc_mbox_mutex);

	/* Read up to 128 bytes at a time, since the CUC command can
	 * only return up to 253 bytes.
	 */
	while (len) {
		u32 to_read = min_t(u32, len, 128);

		uc_prepare_comm(hw);

		hw->uc_req.cmd = CUC_CMD_QSFP_READ;
		hw->uc_req.count = sizeof(struct cuc_qsfp_read_req_data) + 1;
		req_data->nic = CUC_MAC_THIS_NIC;
		req_data->page = page;
		req_data->addr = offset;
		req_data->count = to_read;

		rc = uc_wait_for_response(hw);
		if (!rc) {
			memcpy(data, rsp_data, to_read);
		} else {
			mutex_unlock(&hw->uc_mbox_mutex);
			cxidev_err(cdev,
				   "uC failed to read the cable eeprom page %u offset %u length %u: %d\n",
				   page, offset, to_read, rc);
			return -EIO;
		}

		len -= to_read;
		offset += to_read;
		data += to_read;
	}

	mutex_unlock(&hw->uc_mbox_mutex);

	return ret_len;
}
EXPORT_SYMBOL(cxi_get_qsfp_data);

/* Read the cable module information. Assume it's a SFF-8436 /
 * SFF-8636 type, with 256-bytes eeprom pages.
 */
#define SFF_8436_STATUS_2_OFFSET           0x02
#define     CMIS_PAGE_01_PRESENT           BIT(7)
static void uc_update_cable_info(struct cass_dev *hw)
{
	int ret;

	mutex_lock(&hw->qsfp_eeprom_lock);
	hw->qsfp_eeprom_page_len = 0;
	mutex_unlock(&hw->qsfp_eeprom_lock);

	ret = cxi_get_qsfp_data(&hw->cdev, 0, ETH_MODULE_SFF_8436_LEN,
				0, &hw->qsfp_eeprom_page0[0]);
	if (ret != ETH_MODULE_SFF_8436_LEN)
		return;

	mutex_lock(&hw->qsfp_eeprom_lock);
	hw->qsfp_eeprom_page_len = ETH_MODULE_SFF_8436_LEN;
	mutex_unlock(&hw->qsfp_eeprom_lock);

	if ((hw->qsfp_eeprom_page0[QSFP_IDENTIFIER_OFFSET] >= 0x1E) &&
	    (hw->qsfp_eeprom_page0[QSFP_IDENTIFIER_OFFSET] <= 0x25)) {
		if (((hw->qsfp_eeprom_page0[QSFP_REV_CMPL_OFFSET] & 0xF0) == 0x30) ||
		    ((hw->qsfp_eeprom_page0[QSFP_REV_CMPL_OFFSET] & 0xF0) == 0x40) ||
		    ((hw->qsfp_eeprom_page0[QSFP_REV_CMPL_OFFSET] & 0xF0) == 0x50)) {
			if (hw->qsfp_eeprom_page0[SFF_8436_STATUS_2_OFFSET] & CMIS_PAGE_01_PRESENT) {
				cxidev_dbg(&hw->cdev, "no eeprom page 1\n");
				return;
			}

			ret = cxi_get_qsfp_data(&hw->cdev, 0, ETH_MODULE_SFF_8436_LEN,
						1, &hw->qsfp_eeprom_page1[0]);
			if (ret != ETH_MODULE_SFF_8436_LEN)
				cxidev_dbg(&hw->cdev, "couldn't read eeprom page 1: %d\n", ret);
		}
	}
}

/* Write to the QSFP module - CUC_CMD_QSFP_WRITE */
int uc_cmd_qsfp_write(struct cass_dev *hw, u8 page, u8 addr,
		      const u8 *data, size_t data_len)
{
	struct cuc_qsfp_write_req_data *req_data =
		(struct cuc_qsfp_write_req_data *)hw->uc_req.data;
	int rc;

	if (!hw->uc_present)
		return -EIO;

	if (data_len > ETH_MODULE_SFF_8436_LEN)
		return -EINVAL;

	if (addr + data_len > ETH_MODULE_SFF_8436_LEN)
		return -EINVAL;

	mutex_lock(&hw->uc_mbox_mutex);

	while (data_len) {
		size_t to_write = data_len;
		const size_t max_data_size =
			CUC_DATA_BYTES - sizeof(struct cuc_qsfp_write_req_data);

		/* If the data is too big, write 128 bytes, then the
		 * remainder in a second request.
		 */
		if (to_write > max_data_size)
			to_write = ETH_MODULE_SFF_8436_LEN / 2;

		uc_prepare_comm(hw);

		hw->uc_req.cmd = CUC_CMD_QSFP_WRITE;
		hw->uc_req.count = sizeof(struct cuc_qsfp_write_req_data) +
			to_write + 1;
		req_data->nic = CUC_MAC_THIS_NIC;
		req_data->page = page;
		req_data->addr = addr;
		req_data->count = to_write;
		memcpy(req_data->data, data, to_write);

		rc = uc_wait_for_response(hw);
		if (rc) {
			mutex_unlock(&hw->uc_mbox_mutex);

			cxidev_err(&hw->cdev,
				   "uC failed to write QSFP data: %d\n", rc);
			return rc;
		}

		data += to_write;
		data_len -= to_write;
	}

	mutex_unlock(&hw->uc_mbox_mutex);

	return 0;
}

/* Return the ISR */
static u32 uc_cmd_get_intr_locked(struct cass_dev *hw)
{
	struct cuc_get_intr_req_data *req_data =
		(struct cuc_get_intr_req_data *)hw->uc_req.data;
	const struct cuc_get_intr_rsp_data *rsp_data =
		(struct cuc_get_intr_rsp_data *)hw->uc_resp.data;
	int rc;
	u32 isr;

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_GET_INTR;
	hw->uc_req.count = sizeof(*req_data) + 1;
	req_data->nic = CUC_MAC_THIS_NIC;

	rc = uc_wait_for_response(hw);
	if (!rc)
		isr = rsp_data->isr;
	else
		isr = 0;

	return isr;
}

static u32 uc_cmd_get_intr(struct cass_dev *hw)
{
	u32 isr;

	mutex_lock(&hw->uc_mbox_mutex);
	isr = uc_cmd_get_intr_locked(hw);
	mutex_unlock(&hw->uc_mbox_mutex);

	return isr;
}

/* CUC_CMD_CLEAR_INTR */
static void uc_cmd_clear_intr_locked(struct cass_dev *hw, u32 isr)
{
	struct cuc_clear_isr_req_data *req_data =
		(struct cuc_clear_isr_req_data *)hw->uc_req.data;
	int rc;

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_CLEAR_ISR;
	hw->uc_req.count = sizeof(*req_data) + 1;

	req_data->nic = CUC_MAC_THIS_NIC;
	req_data->isr_clear_bits = isr;

	rc = uc_wait_for_response(hw);
	if (rc)
		cxidev_err(&hw->cdev, "Failed to clear the uC interrupt: %d\n", rc);
}

static void uc_cmd_clear_intr(struct cass_dev *hw, u32 isr)
{
	mutex_lock(&hw->uc_mbox_mutex);
	uc_cmd_clear_intr_locked(hw, isr);
	mutex_unlock(&hw->uc_mbox_mutex);
}

/* CUC_CMD_UPDATE_IER */
void uc_cmd_update_ier(struct cass_dev *hw, u32 set_bits, u32 clear_bits)
{
	struct cuc_update_ier_req_data *req_data =
		(struct cuc_update_ier_req_data *)hw->uc_req.data;
	int rc;

	if (!hw->uc_present)
		return;

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_UPDATE_IER;
	hw->uc_req.count = sizeof(*req_data) + 1;

	req_data->nic = CUC_MAC_THIS_NIC;
	req_data->ier_set_bits = set_bits;
	req_data->ier_clear_bits = clear_bits;

	rc = uc_wait_for_response(hw);
	if (rc)
		cxidev_err(&hw->cdev, "Failed to update the uC IER: %d\n", rc);

	mutex_unlock(&hw->uc_mbox_mutex);
}

void uc_cmd_set_link_leds(struct cass_dev *hw, enum casuc_led_states led_state)
{
	struct cuc_set_led_req *req_data =
		(struct cuc_set_led_req *)hw->uc_req.data;
	int rc;

	if (!hw->uc_present)
		return;

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_SET_LED;
	hw->uc_req.count = sizeof(*req_data) + 1;

	req_data->nic = CUC_MAC_THIS_NIC;
	req_data->led = LED_LINK_STATUS;
	req_data->state = led_state;

	rc = uc_wait_for_response(hw);

	mutex_unlock(&hw->uc_mbox_mutex);

	if (rc)
		cxidev_err(&hw->cdev, "Failed to set the link LED to %d: %d\n",
			   led_state, rc);
}

/**
 * cxi_program_firmware()
 *
 * Flash the uC firmware, reboots it and wait until it's back up. It
 * is expected that the whole procedure would not last more than 30
 * seconds.
 *
 * @cdev: CXI device
 * @fw: a firmware returned by request_firmware()
 *
 * Returns 0 on success, or a negative errno.
 *
 */
int cxi_program_firmware(struct cxi_dev *cdev, const struct firmware *fw)
{
	struct cass_dev *hw = container_of(cdev, struct cass_dev, cdev);
	struct cuc_firmware_update_start_req *req_start =
		(struct cuc_firmware_update_start_req *)hw->uc_req.data;
	struct cuc_firmware_update_download_req *req_download =
		(struct cuc_firmware_update_download_req *)hw->uc_req.data;
	struct cuc_firmware_update_status_rsp *status =
		(struct cuc_firmware_update_status_rsp *)hw->uc_resp.data;
	size_t len_left;
	size_t to_send;
	const u8 *data;
	int rc;
	ktime_t timeout;

	if (!hw->uc_present) {
		cxidev_info(cdev, "uC is not supported\n");
		return -EIO;
	}

	if (!capable(CAP_SYS_ADMIN))
		return -EPERM;

	if (!cdev->is_physfn) {
		cxidev_err(cdev, "Cannot flash Cassini firmware from a VF\n");
		return -EINVAL;
	}

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_FIRMWARE_UPDATE_START;
	hw->uc_req.count = 1 + sizeof(*req_start);

	req_start->nic = CUC_MAC_THIS_NIC;
	req_start->size = fw->size;
	req_start->slot = FW_SLOT_ACTIVE;

	rc = uc_wait_for_response(hw);
	if (rc) {
		cxidev_err(cdev, "Cannot start firmware flashing: %d\n", rc);
		goto done;
	}

	/* Upload the firmware by tiny chunks */
	len_left = fw->size;
	data = fw->data;
	while (len_left) {
		to_send = min_t(size_t, CUC_DATA_BYTES - sizeof(*req_download),
				len_left);

		uc_prepare_comm(hw);

		hw->uc_req.cmd = CUC_CMD_FIRMWARE_UPDATE_DOWNLOAD;
		hw->uc_req.count = 1 + sizeof(*req_download) + to_send;

		memcpy(req_download->data, data, to_send);
		len_left -= to_send;
		data += to_send;

		rc = uc_wait_for_response(hw);
		if (rc) {
			cxidev_err(cdev, "Cannot upload firmware to uC: %d\n",
				   rc);
			goto done;
		}
	}

	/* Wait until idle. Flashing should not take more than 30
	 * seconds. Set the timeout to 130 seconds.
	 */
	timeout = ktime_add_ms(ktime_get(), 130 * MSEC_PER_SEC);
	while (1) {
		uc_prepare_comm(hw);

		hw->uc_req.cmd = CUC_CMD_FIRMWARE_UPDATE_STATUS;
		hw->uc_req.count = 1;

		rc = uc_wait_for_response(hw);
		if (rc) {
			cxidev_err(cdev,
				   "Cannot get firmware status from uC: %d\n",
				   rc);
			goto done;
		}

		if (status->status & FWU_STATUS_IDLE) {
			if (status->status != FWU_STATUS_SUCCESS) {
				cxidev_err(cdev,
					   "Firmware flashing failed: 0x%x\n",
					   status->status);
				rc = -EIO;
				goto done;
			} else {
				cxidev_info(cdev,
					    "Firmware flashing succeeded\n");
				break;
			}
		}

		if (ktime_after(ktime_get(), timeout)) {
			cxidev_err(cdev,
				   "uC took too long to flash the new firmware\n");
			rc = -EIO;
			goto done;
		}

		mdelay(100);
	}

	/* Clear the reset bit just in case */
	uc_cmd_clear_intr_locked(hw, ATT1_UC_RESET);

	/* Reset the uC */
	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_RESET;
	hw->uc_req.count = 1;

	rc = uc_wait_for_response(hw);
	if (rc) {
		cxidev_err(cdev, "uC didn't want to reset after flashing: %d\n",
			   rc);
		rc = -EIO;
		goto done;
	}

	/* Poll the uC until the reset bit is set. Reset should take
	 * about 2 seconds. Set the timeout to 20 seconds.
	 */
	timeout = ktime_add_ms(ktime_get(), 20 * MSEC_PER_SEC);
	while (1) {
		u32 isr;

		isr = uc_cmd_get_intr_locked(hw);

		if (isr & ATT1_UC_RESET)
			break;

		if (ktime_after(ktime_get(), timeout)) {
			rc = -ETIMEDOUT;
			cxidev_err(cdev, "uC took too long to reset\n");
			goto done;
		}

		mdelay(100);
	}

	rc = 0;

done:
	mutex_unlock(&hw->uc_mbox_mutex);

	if (!rc)
		uc_cmd_get_fw_versions(hw);

	return rc;
}
EXPORT_SYMBOL(cxi_program_firmware);

/* Called when one of the C_PI_ERR_FLG.UC_ATTENTION bits was set */
static void uc_attention_cb(struct cass_dev *hw, unsigned int irq,
			    bool is_ext, unsigned int bitn)
{
	if (bitn == C_PI_ERR_FLG__UC_ATTENTION_0)
		complete(&hw->uc_attention0_comp);

	if (bitn == C_PI_ERR_FLG__UC_ATTENTION_1)
		queue_work(hw->uc_attn1_wq, &hw->uc_attn1_work);
}

static ssize_t show_fw_version(struct kobject *kobj, char *buf,
			       unsigned int fw_target)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, uc_kobj);
	int rc;

	mutex_lock(&hw->uc_mbox_mutex);

	if (hw->fw_versions[fw_target])
		rc = scnprintf(buf, PAGE_SIZE, "%s\n",
			       hw->fw_versions[fw_target]);
	else
		rc = 0;

	mutex_unlock(&hw->uc_mbox_mutex);

	return rc;
}

#define FW_VERSION_ATTR(name, target_fw) \
	static ssize_t name##_show(struct kobject *kobj, \
				  struct kobj_attribute *kattr, char *buf) \
	{								\
		return show_fw_version(kobj, buf, target_fw);		\
	}								\
	static struct kobj_attribute dev_attr_##name = __ATTR_RO(name)

FW_VERSION_ATTR(app_version, FW_UC_APPLICATION);
FW_VERSION_ATTR(bootloader_version, FW_UC_BOOTLOADER);
FW_VERSION_ATTR(qspi_blob_version, FW_QSPI_BLOB);
FW_VERSION_ATTR(oprom_version, FW_OPROM);
FW_VERSION_ATTR(csr1_version, FW_CSR1);
FW_VERSION_ATTR(csr2_version, FW_CSR2);
FW_VERSION_ATTR(srds_version, FW_SRDS);

/* Reset the uC through sysfs. As a safety measure, the user must
 * write '1789' to it.
 */
static ssize_t reset_store(struct kobject *kobj, struct kobj_attribute *attr,
			   const char *buf, size_t len)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, uc_kobj);
	int rc;
	int value;

	rc = kstrtoint(buf, 0, &value);
	if (rc)
		return rc;

	if (value != 1789)
		return -EINVAL;

	if (!hw->uc_present)
		return 0;

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_RESET;
	hw->uc_req.count = 1;

	rc = uc_wait_for_response(hw);
	if (rc)
		cxidev_err(&hw->cdev, "uC didn't want to reset: %d\n", rc);
	else
		rc = len;

	mutex_unlock(&hw->uc_mbox_mutex);

	return rc;
}

static struct kobj_attribute dev_attr_reset = __ATTR_WO(reset);

static ssize_t timings_show(struct kobject *kobj,
			    struct kobj_attribute *kattr, char *buf)
{
	struct cass_dev *hw =
		container_of(kobj, struct cass_dev, uc_kobj);
	const struct cuc_get_timings_rsp *rsp_data =
		(struct cuc_get_timings_rsp *)hw->uc_resp.data;
	int rc;
	int i;

	if (!hw->uc_present)
		return -ENODATA;

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_GET_TIMINGS;
	hw->uc_req.count = 1;

	rc = uc_wait_for_response(hw);
	if (rc) {
		cxidev_err(&hw->cdev, "uC didn't return its timings: %d\n", rc);
	} else {
		rc = 0;
		for (i = 0; i < TIMING_NUM_ENTRIES; i++) {
			rc += scnprintf(buf + rc, PAGE_SIZE - rc, "%llu ",
					rsp_data->entries_us[i]);
		}
		rc += scnprintf(buf + rc, PAGE_SIZE - rc, "\n");
	}

	mutex_unlock(&hw->uc_mbox_mutex);

	return rc;
}

static struct kobj_attribute dev_attr_timings = __ATTR_RO(timings);

static struct attribute *uc_attrs[] = {
	&dev_attr_app_version.attr,
	&dev_attr_bootloader_version.attr,
	&dev_attr_qspi_blob_version.attr,
	&dev_attr_oprom_version.attr,
	&dev_attr_csr1_version.attr,
	&dev_attr_csr2_version.attr,
	&dev_attr_srds_version.attr,
	&dev_attr_reset.attr,
	&dev_attr_timings.attr,
	NULL,
};
ATTRIBUTE_GROUPS(uc);

static struct kobj_type uc_sysfs_entries = {
	.sysfs_ops      = &kobj_sysfs_ops,
	.default_groups = uc_groups,
};

/* Retrieve a PDR given a record handle.
 * Used for PDR iteration.
 *
 * The uC must be locked.
 */
static int pldm_get_pdr(struct cass_dev *hw, unsigned int record_handle)
{
	struct get_pdr_req *req_data =
		(struct get_pdr_req *)hw->uc_req.data;
	int rc;

	if (!hw->uc_present)
		return -ENOTSUPP;

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_PLDM;
	hw->uc_req.count = sizeof(*req_data) + 1;

	req_data->hdr.d = 0;
	req_data->hdr.rq = 1;
	req_data->hdr.pldm_type = PLDM_TYPE_PLATFORM_MONITORING_AND_CONTROL;
	req_data->hdr.hdr_ver = 0;
	req_data->hdr.pldm_command_code = PLDM_CMD_GET_PDR;
	req_data->record_handle = record_handle;
	req_data->data_transfer_handle = 0;
	req_data->transfer_operation_flag = PLDM_XFER_OP_GET_FIRST_PART;
	req_data->request_count = CUC_DATA_BYTES - sizeof(struct get_pdr_rsp);
	req_data->record_change_number = 0;

	rc = uc_wait_for_response(hw);

	return rc;
}

/* Iterate through the PDR building an IDR tree of supported
 * sensors.
 */
static void iterate_pdr(struct cass_dev *hw)
{
	unsigned int record_handle;
	const struct get_pdr_rsp *rsp_data =
		(struct get_pdr_rsp *)hw->uc_resp.data;
	const struct pdr_hdr *pdr_hdr = (struct pdr_hdr *)rsp_data->record_data;
	int rc;
	struct pldm_sensor *sensor;
	unsigned int count = 0;

	mutex_lock(&hw->uc_mbox_mutex);

	/* Get the numerical sensors first */
	record_handle = 0;
	do {
		const struct numeric_sensor_pdr *pdr_sensor =
			(struct numeric_sensor_pdr *)rsp_data->record_data;

		rc = pldm_get_pdr(hw, record_handle);
		if (rc)
			break;

		if (rsp_data->completion_code != PLDM_SUCCESS) {
			cxidev_err(&hw->cdev,
				   "Unexpected PLDM completion error: %d\n",
				   rsp_data->completion_code);
			break;
		}

		record_handle = rsp_data->next_record_handle;

		if (pdr_hdr->pdr_type != PLDM_PDR_NUMERIC_SENSOR)
			continue;

		sensor = kzalloc(sizeof(struct pldm_sensor), GFP_KERNEL);
		if (!sensor)
			break;

		sensor->hw = hw;
		sensor->id = pdr_sensor->sensor_id;
		sensor->num = *pdr_sensor;

		idr_preload(GFP_KERNEL);
		rc = idr_alloc(&hw->pldm_sensors, sensor,
			       sensor->num.sensor_id, sensor->num.sensor_id + 1,
			       GFP_NOWAIT);
		idr_preload_end();

		count++;
	} while (record_handle != 0);

	/* Then get their names */
	record_handle = 0;
	do {
		const struct aux_name_pdr *pdr_name =
			(struct aux_name_pdr *)rsp_data->record_data;
		int i;

		rc = pldm_get_pdr(hw, record_handle);
		if (rc)
			break;

		if (rsp_data->completion_code != PLDM_SUCCESS) {
			cxidev_err(&hw->cdev,
				   "Unexpected PLDM completion error: %d\n",
				   rsp_data->completion_code);
			break;
		}

		record_handle = rsp_data->next_record_handle;

		if (pdr_hdr->pdr_type != PLDM_PDR_SENSOR_AUXILIARY_NAMES)
			continue;

		sensor = idr_find(&hw->pldm_sensors, pdr_name->sensor_id);
		if (!sensor)
			continue;

		if (pdr_name->name_string_count != 1) {
			cxidev_err(&hw->cdev,
				   "Unexpected number of strings: %d\n",
				   pdr_name->name_string_count);
			continue;
		}

		/* Name is in big endian UTF-16. Convert to ASCII. */
		for (i = 0; i < AUX_NAME_MAX; i++) {
			u16 c = be16_to_cpu((__force __be16)pdr_name->sensor_name[i]);

			if (c == 0)
				break;

			if (c < ' ' || c > '~')
				c = '.';

			sensor->name[i] = c;
		}
		sensor->name[AUX_NAME_MAX - 1] = 0;
	} while (record_handle != 0);

	mutex_unlock(&hw->uc_mbox_mutex);

	hw->pldm_sensors_count = count;
}

/* Read a PLDM sensor, given a sensor ID.
 *
 * uc_mbox_mutex must be held.
 *
 * Return 0 if the sensor was successfully read.
 */
static int pldm_read_sensor(struct cass_dev *hw, unsigned int id,
			    struct pldm_sensor_reading *result)
{
	struct get_sensor_reading_req *req_data =
		(struct get_sensor_reading_req *)hw->uc_req.data;
	const struct get_sensor_reading_rsp *rsp_data =
		(struct get_sensor_reading_rsp *)hw->uc_resp.data;
	int rc;

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_PLDM;
	hw->uc_req.count = sizeof(*req_data) + 1;

	req_data->hdr.d = 0;
	req_data->hdr.rq = 1;
	req_data->hdr.pldm_type = PLDM_TYPE_PLATFORM_MONITORING_AND_CONTROL;
	req_data->hdr.hdr_ver = 0;
	req_data->hdr.pldm_command_code = PLDM_CMD_GET_SENSOR_READING;
	req_data->sensor_id = id;
	req_data->rearm_event_status = 0;

	rc = uc_wait_for_response(hw);
	if (rc)
		goto out;

	if (rsp_data->completion_code != PLDM_SUCCESS) {
		rc = -EIO;
		goto out;
	}

	result->operational_state = rsp_data->sensor_operational_state;
	result->present_state = rsp_data->present_state;
	result->previous_state = rsp_data->previous_state;
	rc = 0;
	switch (rsp_data->sensor_data_size) {
	case PLDM_DATA_SIZE_UINT8:
		result->present_reading = rsp_data->present_reading.value_UINT8;
		break;
	case PLDM_DATA_SIZE_SINT8:
		result->present_reading = rsp_data->present_reading.value_SINT8;
		break;
	case PLDM_DATA_SIZE_UINT16:
		result->present_reading = rsp_data->present_reading.value_UINT16;
		break;
	case PLDM_DATA_SIZE_SINT16:
		result->present_reading = rsp_data->present_reading.value_SINT16;
		break;
	case PLDM_DATA_SIZE_UINT32:
		result->present_reading = rsp_data->present_reading.value_UINT32;
		break;
	case PLDM_DATA_SIZE_SINT32:
		result->present_reading = rsp_data->present_reading.value_SINT32;
		break;
	default:
		rc = -EIO;
	}

out:
	return rc;
}

int update_sensor(struct pldm_sensor *sensor,
		  struct pldm_sensor_reading *result)
{
	struct cass_dev *hw = sensor->hw;
	int rc;

	// TODO: rate limit ?
	mutex_lock(&hw->uc_mbox_mutex);

	rc = pldm_read_sensor(hw, sensor->id, result);

	mutex_unlock(&hw->uc_mbox_mutex);

	return rc;
}

static void set_uc_platform(struct cass_dev *hw)
{
	const struct cuc_board_info_rsp *rsp_data =
		(struct cuc_board_info_rsp *)hw->uc_resp.data;
	u8 board_type;
	int rc;
	int resp_len;

	if (HW_PLATFORM_Z1(hw)) {
		if (!cass_version(hw, CASSINI_1))
			hw->uc_platform = CUC_BOARD_TYPE_WASHINGTON;
		else
			hw->uc_platform = CUC_BOARD_TYPE_SAWTOOTH;
		return;
	}

	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_BOARD_INFO;
	hw->uc_req.count = 1;

	rc = uc_wait_for_response(hw);
	board_type = rsp_data->board_type;
	resp_len = hw->uc_resp.count;

	mutex_unlock(&hw->uc_mbox_mutex);

	/* CUC_CMD_BOARD_INFO re-used the CUC_CMD_FW_VERSION command
	 * number. To ensure that the response is indeed for the
	 * CUC_CMD_BOARD_INFO command, also check the response size. A
	 * CUC_CMD_FW_VERSION response would be at least 4 bytes.
	 */
	if (rc == 0 && resp_len == (1 + sizeof(struct cuc_board_info_rsp))) {
		hw->uc_platform = board_type;

		return;
	}

	/* Older firmwares do not support the BOARD_INFO command, and
	 * FRU has to be parsed.
	 */

	if (!hw->fru_info[PLDM_FRU_FIELD_PART_NUMBER]) {
		cxidev_err(&hw->cdev, "Failed to detect platform from uC!\n");
		hw->uc_platform = CUC_BOARD_TYPE_UNKNOWN;
	} else if (!strcmp(hw->fru_info[PLDM_FRU_FIELD_PART_NUMBER],
			   PCA_SHASTA_SAWTOOTH_PASS_1_PN) ||
		   !strcmp(hw->fru_info[PLDM_FRU_FIELD_PART_NUMBER],
			   PCA_SHASTA_SAWTOOTH_PASS_2_PN) ||
		   !strncmp(hw->fru_info[PLDM_FRU_FIELD_PART_NUMBER],
			    PCA_SHASTA_SAWTOOTH_PASS_3_PN,
			    strlen(PCA_SHASTA_SAWTOOTH_PASS_3_PN))) {
		hw->uc_platform = CUC_BOARD_TYPE_SAWTOOTH;
	} else if (!strcmp(hw->fru_info[PLDM_FRU_FIELD_PART_NUMBER],
			   PCA_SHASTA_BRAZOS_PASS_1_PN) ||
		   !strcmp(hw->fru_info[PLDM_FRU_FIELD_PART_NUMBER],
			   PCA_SHASTA_BRAZOS_PASS_2_PN) ||
		   !strcmp(hw->fru_info[PLDM_FRU_FIELD_PART_NUMBER],
			   PCA_SHASTA_BRAZOS_PASS_3_PN) ||
		   !strncmp(hw->fru_info[PLDM_FRU_FIELD_PART_NUMBER],
			    PCA_SHASTA_BRAZOS_PASS_4_PN,
			    strlen(PCA_SHASTA_BRAZOS_PASS_4_PN))) {
		hw->uc_platform = CUC_BOARD_TYPE_BRAZOS;
	} else {
		hw->uc_platform = CUC_BOARD_TYPE_UNKNOWN;
	};
}

static void check_sensor_values(struct cass_dev *hw)
{
	struct pldm_sensor *sensor;
	struct pldm_sensor_reading result;
	int id;
	int rc;

	mutex_lock(&hw->uc_mbox_mutex);

	idr_for_each_entry(&hw->pldm_sensors, sensor, id) {
		unsigned int thresh_mask;

		rc = pldm_read_sensor(hw, id, &result);
		if (rc)
			continue;

		if (result.operational_state != PLDM_OPSTATE_ENABLED)
			continue;

		thresh_mask = get_pldm_value(sensor, supported_thresholds);

		if (((thresh_mask & PLDM_THRESHOLD_LOWER_FATAL_MASK) &&
		     result.present_reading <= get_pldm_value(sensor, fatal_low)) ||
		    ((thresh_mask & PLDM_THRESHOLD_UPPER_FATAL_MASK) &&
		     result.present_reading >= get_pldm_value(sensor, fatal_high))) {
			cxidev_err(&hw->cdev,
				   "uC: ATT1_SENSOR_ALERT: sensor '%s' (%d) in fatal state: %lld not in [%lld..%lld]\n",
				   sensor->name, id,
				   result.present_reading,
				   get_pldm_value(sensor, fatal_low),
				   get_pldm_value(sensor, fatal_high));
		} else if (((thresh_mask & PLDM_THRESHOLD_LOWER_CRITICAL_MASK) &&
			  result.present_reading <= get_pldm_value(sensor, critical_low)) ||
			 ((thresh_mask & PLDM_THRESHOLD_UPPER_CRITICAL_MASK) &&
			  result.present_reading >= get_pldm_value(sensor, critical_high))) {
			cxidev_warn(&hw->cdev,
				    "uC: ATT1_SENSOR_ALERT: sensor '%s' (%d) in critical state: %lld not in [%lld..%lld]\n",
				    sensor->name, id,
				    result.present_reading,
				    get_pldm_value(sensor, critical_low),
				    get_pldm_value(sensor, critical_high));
		} else if (((thresh_mask & PLDM_THRESHOLD_LOWER_WARNING_MASK) &&
			  result.present_reading <= get_pldm_value(sensor, warning_low)) ||
			 ((thresh_mask & PLDM_THRESHOLD_UPPER_WARNING_MASK) &&
			  result.present_reading >= get_pldm_value(sensor, warning_high))) {
			cxidev_warn(&hw->cdev,
				    "uC: ATT1_SENSOR_ALERT: sensor '%s' (%d) in warning state: %lld not in [%lld..%lld]\n",
				    sensor->name, id,
				    result.present_reading,
				    get_pldm_value(sensor, warning_low),
				    get_pldm_value(sensor, warning_high));
		}
	}

	mutex_unlock(&hw->uc_mbox_mutex);
}

/* Called when the µC raised C_PI_ERR_FLG.UC_ATTENTION[1]. Retrieve
 * the status register and act on the bits that are set.
 */
static void attn1_worker(struct work_struct *work)
{
	struct cass_dev *hw =
		container_of(work, struct cass_dev, uc_attn1_work);
	u32 isr;

	/* Get the current interrupt status */
	isr = uc_cmd_get_intr(hw);

	if (isr & ATT1_UC_RESET)
		cxidev_info(&hw->cdev, "uC: UC_RESET\n");

	if (isr & ATT1_ASIC_PWR_UP_DONE)
		cxidev_dbg(&hw->cdev, "uC: ASIC_PWR_UP_DONE\n");

	if (isr & ATT1_ASIC_EPO_TEMPERATURE)
		cxidev_notice(&hw->cdev, "uC: ASIC_EPO_TEMPERATURE\n");

	/* The ATT1_QSFP_BAD_CABLE event happens every 5
	 * seconds. Ignore all but the first one.
	 */
	if (isr & ATT1_QSFP_BAD_CABLE && !hw->qsfp_bad) {
		cxidev_warn(&hw->cdev, "Bad cable present\n");
		hw->qsfp_bad = true;
	}

	if (isr & ATT1_QSFP_INSERT) {
		cxidev_info(&hw->cdev, "Cable inserted\n");
		hw->qsfp_bad = false;
		uc_update_cable_info(hw);
		cass_cable_scan(hw);
		cxi_send_async_event(&hw->cdev, CXI_EVENT_CABLE_INSERTED);
		cass_phy_trigger_machine(hw);
	}

	if (isr & ATT1_QSFP_REMOVE) {
		cxidev_info(&hw->cdev, "Cable removed\n");
		hw->qsfp_bad = false;
		mutex_lock(&hw->qsfp_eeprom_lock);
		hw->qsfp_eeprom_page_len = 0;
		mutex_unlock(&hw->qsfp_eeprom_lock);
		cass_cable_scan(hw);
		cxi_send_async_event(&hw->cdev, CXI_EVENT_CABLE_REMOVED);
		cass_phy_trigger_machine(hw);
	}

	if (isr & ATT1_QSFP_EPO_TEMPERATURE) {
		cxidev_crit(&hw->cdev, "uC: QSFP_EPO_TEMPERATURE\n");
		hw->qsfp_over_temp = true;
		cass_link_set_led(hw);
	}

	if (isr & ATT1_SENSOR_ALERT) {
		cxidev_info(&hw->cdev, "uC: ATT1_SENSOR_ALERT\n");
		check_sensor_values(hw);
	}

	uc_cmd_clear_intr(hw, isr);
}

/* Conversion table for unit between PLDM and HWMON */
static const struct {
	enum sensor_unit pldm_base_unit;
	enum hwmon_sensor_types hwmon_type;
	unsigned int hwmon_input;
} pldm_sensor_to_hwmon[] = {
	{ PLDM_UNIT_DEGREES_C, hwmon_temp, HWMON_T_INPUT | HWMON_T_LABEL, },
	{ PLDM_UNIT_VOLTS, hwmon_in, HWMON_I_INPUT | HWMON_I_LABEL, },
	{ PLDM_UNIT_AMPS, hwmon_curr, HWMON_C_INPUT | HWMON_C_LABEL, },
	{ PLDM_UNIT_WATTS, hwmon_power, HWMON_P_INPUT | HWMON_P_LABEL, },
};

static umode_t cxi_hwmon_is_visible(const void *data,
				    enum hwmon_sensor_types type,
				    u32 attr, int channel)
{
	return 0444;
}

static int cxi_hwmon_read(struct device *dev, enum hwmon_sensor_types type,
			  u32 attr, int channel, long *val)
{
	struct cass_dev *hw = dev_get_drvdata(dev);
	struct pldm_sensor_reading result;
	struct pldm_sensor *sensor;
	bool is_input = false;
	bool found = false;
	int id;
	int rc;

	idr_for_each_entry(&hw->pldm_sensors, sensor, id) {
		if (sensor->hwmon.type == type &&
		    sensor->hwmon.channel == channel) {
			found = true;
			break;
		}
	}

	if (!found)
		return -EINVAL;

	switch (type) {
	case hwmon_temp:
		switch (attr) {
		case hwmon_temp_input:
			is_input = true;
			break;
		case hwmon_temp_max:
			*val = get_pldm_value(sensor, warning_high);
			break;
		case hwmon_temp_crit:
			*val = get_pldm_value(sensor, critical_high);
			break;
		default:
			return -EOPNOTSUPP;
		}
		break;

	case hwmon_in:
		switch (attr) {
		case hwmon_in_input:
			is_input = true;
			break;
		case hwmon_in_min:
			*val = get_pldm_value(sensor, warning_low);
			break;
		case hwmon_in_max:
			*val = get_pldm_value(sensor, warning_high);
			break;
		case hwmon_in_lcrit:
			*val = get_pldm_value(sensor, critical_low);
			break;
		case hwmon_in_crit:
			*val = get_pldm_value(sensor, critical_high);
			break;
		default:
			return -EOPNOTSUPP;
		}
		break;

	case hwmon_curr:
		switch (attr) {
		case hwmon_curr_input:
			is_input = true;
			break;
		case hwmon_curr_max:
			*val = get_pldm_value(sensor, warning_high);
			break;
		case hwmon_curr_crit:
			*val = get_pldm_value(sensor, critical_high);
			break;
		default:
			return -EOPNOTSUPP;
		}
		break;

	case hwmon_power:
		switch (attr) {
		case hwmon_power_input:
			is_input = true;
			break;
		case hwmon_power_max:
			*val = get_pldm_value(sensor, warning_high);
			break;
		case hwmon_power_crit:
			*val = get_pldm_value(sensor, critical_high);
			break;
		default:
			return -EOPNOTSUPP;
		}
		break;

	default:
		return -EOPNOTSUPP;
	}

	if (is_input) {
		rc = update_sensor(sensor, &result);
		if (rc)
			return rc;

		if (result.operational_state == PLDM_OPSTATE_ENABLED)
			*val = result.present_reading;
		else
			return -ENODATA;
	}

	*val *= sensor->hwmon.multiplier;

	return 0;
}

static int cxi_hwmon_read_string(struct device *dev,
				 enum hwmon_sensor_types type,
				 u32 attr, int channel, const char **str)
{
	struct cass_dev *hw = dev_get_drvdata(dev);
	struct pldm_sensor *sensor;
	int id;

	idr_for_each_entry(&hw->pldm_sensors, sensor, id) {
		if (sensor->hwmon.type == type &&
		    sensor->hwmon.channel == channel) {
			*str = sensor->name;
			return 0;
		}
	}

	*str = "NOTFOUND";

	return 0;
}

static const struct hwmon_ops cxi_hwmon_ops = {
	.is_visible = cxi_hwmon_is_visible,
	.read = cxi_hwmon_read,
	.read_string = cxi_hwmon_read_string,
};

static const long pldm_multiplier[] = {
	1000000000,		/* PLDM_MODIFIER_NANO */
	100000000,
	10000000,
	1000000,		/* PLDM_MODIFIER_MICRO */
	100000,
	10000,
	1000,			/* PLDM_MODIFIER_MILLI */
	100,
	10,			/* PLDM_MODIFIER_DECI */
	1,			/* PLDM_MODIFIER_NONE */
};

static const unsigned int pldm_to_hwmon_temp[PLDM_THRESHOLD_COUNT] = {
	[PLDM_THRESHOLD_UPPER_WARNING] = HWMON_T_MAX,
	[PLDM_THRESHOLD_UPPER_CRITICAL] = HWMON_T_CRIT,
};

static const unsigned int pldm_to_hwmon_in[PLDM_THRESHOLD_COUNT] = {
	[PLDM_THRESHOLD_UPPER_WARNING] = HWMON_I_MAX,
	[PLDM_THRESHOLD_LOWER_WARNING] = HWMON_I_MIN,
	[PLDM_THRESHOLD_LOWER_CRITICAL] = HWMON_I_LCRIT,
	[PLDM_THRESHOLD_UPPER_CRITICAL] = HWMON_I_CRIT,
};

static const unsigned int pldm_to_hwmon_curr[PLDM_THRESHOLD_COUNT] = {
	[PLDM_THRESHOLD_UPPER_WARNING] = HWMON_C_MAX,
	[PLDM_THRESHOLD_UPPER_CRITICAL] = HWMON_C_CRIT,
};

static const unsigned int pldm_to_hwmon_power[PLDM_THRESHOLD_COUNT] = {
	[PLDM_THRESHOLD_UPPER_WARNING] = HWMON_P_MAX,
	[PLDM_THRESHOLD_UPPER_CRITICAL] = HWMON_P_CRIT,
};

static void create_sensors_intf(struct cass_dev *hw)
{
	unsigned int config_idx = 0;
	unsigned int sensor_idx = 0;
	unsigned int sensor_type;
	unsigned int channel;

	if (hw->pldm_sensors_count > CASS_MAX_SENSORS) {
		cxidev_WARN_ONCE(&hw->cdev, true, "Too many sensors for hwmon\n");
		return;
	}

	/* Enumerate all sensors that can be converted between PLDM
	 * and HWMON, and build a HWMON device.
	 */
	for (sensor_type = 0; sensor_type < ARRAY_SIZE(pldm_sensor_to_hwmon);
	     sensor_type++) {
		struct pldm_sensor *sensor;
		bool present = false;
		int unit_modifier;
		int id;

		/* Find whether a sensor of that PLDM type is present */
		idr_for_each_entry(&hw->pldm_sensors, sensor, id) {
			if (sensor->num.base_unit !=
			    pldm_sensor_to_hwmon[sensor_type].pldm_base_unit)
				continue;

			present = true;
			break;
		}

		if (!present)
			continue;

		hw->hwmon.info[sensor_idx].type =
			pldm_sensor_to_hwmon[sensor_type].hwmon_type;
		hw->hwmon.info[sensor_idx].config =
			(const u32 *)&hw->hwmon.config[config_idx];

		/* Create the config for all sensors of that type */
		channel = 0;
		idr_for_each_entry(&hw->pldm_sensors, sensor, id) {
			unsigned long thresh_mask;
			unsigned int hwmon_input;
			unsigned int bit;

			if (sensor->num.base_unit !=
			    pldm_sensor_to_hwmon[sensor_type].pldm_base_unit)
				continue;

			sensor->hwmon.type =
				pldm_sensor_to_hwmon[sensor_type].hwmon_type;
			sensor->hwmon.channel = channel;

			/* Compute the multiplier to apply to convert
			 * from PLDM value to hwmon, taking into
			 * account both the pldm and hwmon
			 * multipliers.
			 */
			unit_modifier = sensor->num.unit_modifier;
			if (unit_modifier < PLDM_MODIFIER_NANO ||
			    unit_modifier > PLDM_MODIFIER_NONE)
				unit_modifier = PLDM_MODIFIER_NONE;

			unit_modifier -= PLDM_MODIFIER_NANO;

			hwmon_input = pldm_sensor_to_hwmon[sensor_type].hwmon_input;
			thresh_mask = get_pldm_value(sensor, supported_thresholds);

			switch (sensor->hwmon.type) {
			case hwmon_temp:
				sensor->hwmon.multiplier = 1000 /
					pldm_multiplier[unit_modifier];

				for_each_set_bit(bit, &thresh_mask, PLDM_THRESHOLD_COUNT)
					hwmon_input |= pldm_to_hwmon_temp[bit];

				break;
			case hwmon_in:
				sensor->hwmon.multiplier = 1000 /
					pldm_multiplier[unit_modifier];

				for_each_set_bit(bit, &thresh_mask, PLDM_THRESHOLD_COUNT)
					hwmon_input |= pldm_to_hwmon_in[bit];

				break;
			case hwmon_curr:
				sensor->hwmon.multiplier = 1000 /
					pldm_multiplier[unit_modifier];

				for_each_set_bit(bit, &thresh_mask, PLDM_THRESHOLD_COUNT)
					hwmon_input |= pldm_to_hwmon_curr[bit];

				break;
			case hwmon_power:
				sensor->hwmon.multiplier = 1000000 /
					pldm_multiplier[unit_modifier];

				for_each_set_bit(bit, &thresh_mask, PLDM_THRESHOLD_COUNT)
					hwmon_input |= pldm_to_hwmon_power[bit];

				break;
			default:
				sensor->hwmon.multiplier = 1;
			}

			hw->hwmon.config[config_idx] = hwmon_input;

			config_idx++;
			channel++;
		};

		config_idx++;		/* NUL terminator */

		hw->hwmon.all_types[sensor_idx] = &hw->hwmon.info[sensor_idx];
		sensor_idx++;
	}

	hw->hwmon.all_types[sensor_idx] = NULL;

	hw->hwmon.chip_info.ops = &cxi_hwmon_ops;
	hw->hwmon.chip_info.info = hw->hwmon.all_types;

	hw->hwmon.dev = hwmon_device_register_with_info(&hw->cdev.pdev->dev,
							hw->cdev.name,
							hw,
							&hw->hwmon.chip_info,
							NULL);
	if (IS_ERR(hw->hwmon.dev)) {
		dev_err(&hw->cdev.pdev->dev,
			"Failed to register sensors: %ld\n",
			PTR_ERR(hw->hwmon.dev));
		hw->hwmon.dev = NULL;
	}
}

int cass_register_uc(struct cass_dev *hw)
{
	int rc;

	mutex_init(&hw->uc_mbox_mutex);
	init_completion(&hw->uc_attention0_comp);
	INIT_WORK(&hw->uc_attn1_work, attn1_worker);
	idr_init(&hw->pldm_sensors);

	rc = cxi_dmac_desc_set_reserve(&hw->cdev, 32, HOST_TO_UC_FIRST, "uc");
	if (rc < 0) {
		cxidev_err(&hw->cdev, "Failed to reserve UC DMAC space\n");
		return rc;
	}
	hw->uc_dmac_id = rc;

	hw->uc_attn1_wq = alloc_workqueue("uc_attn1", WQ_UNBOUND, 1);
	if (!hw->uc_attn1_wq) {
		cxidev_err(&hw->cdev, "Failed to allocate UC attn1 workqueue\n");
		rc = -ENOMEM;
		goto free_dmac_id;
	}

	/* Create the uC sysfs entries */
	rc = kobject_init_and_add(&hw->uc_kobj, &uc_sysfs_entries,
				  &hw->cdev.pdev->dev.kobj,
				  "uc");
	if (rc < 0)
		goto free_wq;

	/* Register callback for C_PI_ERR_FLG.UC_ATTENTION bits */
	hw->uc_err.irq = C_PI_IRQA_MSIX_INT;
	hw->uc_err.is_ext = false;
	hw->uc_err.err_flags.pi.uc_attention = 0b11;
	hw->uc_err.cb = uc_attention_cb;
	cxi_register_hw_errors(hw, &hw->uc_err);

	/* Some test platforms do not have a uC */
	mutex_lock(&hw->uc_mbox_mutex);

	uc_prepare_comm(hw);

	hw->uc_req.cmd = CUC_CMD_PING;
	hw->uc_req.count = 1;

	rc = uc_wait_for_response(hw);

	hw->uc_present = rc == 0;

	mutex_unlock(&hw->uc_mbox_mutex);

	if (!hw->uc_present)
		cxidev_err(&hw->cdev, "uC not detected: %d\n", rc);

	uc_cmd_update_ier(hw, HOST_CLEARED_ATT1_INTERRUPTS, 0);

	uc_cmd_get_fw_versions(hw);

	/* Find all sensors, and check whether they are currently
	 * reporting some error.
	 */
	iterate_pdr(hw);
	check_sensor_values(hw);

	create_sensors_intf(hw);

	uc_cmd_get_fru(hw);

	set_uc_platform(hw);

	uc_update_cable_info(hw);

	return 0;

free_wq:
	kobject_put(&hw->uc_kobj);
	destroy_workqueue(hw->uc_attn1_wq);

free_dmac_id:
	cxi_dmac_desc_set_free(&hw->cdev, hw->uc_dmac_id);

	return rc;
}

void cass_unregister_uc(struct cass_dev *hw)
{
	struct pldm_sensor *sensor;
	int id;

	uc_cmd_update_ier(hw, 0, ATT1_ALL_INTERRUPTS);
	if (hw->hwmon.dev)
		hwmon_device_unregister(hw->hwmon.dev);
	kobject_put(&hw->uc_kobj);
	cxi_dmac_desc_set_free(&hw->cdev, hw->uc_dmac_id);
	destroy_workqueue(hw->uc_attn1_wq);
	cxi_unregister_hw_errors(hw, &hw->uc_err);
	idr_for_each_entry(&hw->pldm_sensors, sensor, id)
		kfree(sensor);
	idr_destroy(&hw->pldm_sensors);
}
