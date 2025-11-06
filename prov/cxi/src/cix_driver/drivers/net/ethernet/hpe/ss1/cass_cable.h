/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2021 Hewlett Packard Enterprise Development LP */
#ifndef _CASS_CABLE_H
#define _CASS_CABLE_H

#define QSFP_IDENTIFIER_OFFSET                   0
#define QSFP_REV_CMPL_OFFSET                     1
#define QSFP_VENDOR_MAX_STR_LEN                  16

#define QDD_VENDOR_TE                   "TE Connectivity"
#define QDD_VENDOR_HISENSE              "Hisense"
#define QDD_VENDOR_FIT                  "FIT HON TENG"
#define QDD_VENDOR_FT                   "FT HON TENG"
#define QDD_VENDOR_MELLANOX             "Mellanox"
#define QDD_VENDOR_MOLEX                "Molex"
#define QDD_VENDOR_LEONI                "LEONI"
#define QDD_VENDOR_HITACHI              "Hitachi Metals"
#define QDD_VENDOR_LUXSHARE             "LUXSHARE-ICT"
#define QDD_VENDOR_DUST_PHOTONICS       "DustPhotonics"
#define QDD_VENDOR_FINISAR              "FINISAR CORP"
#define QDD_VENDOR_HPE                  "HPE"
#define QDD_VENDOR_CLOUD_LIGHT          "Cloud Light"
#define QDD_VENDOR_AMPHENOL             "Amphenol"

// SFF-8024 Specification for SFF Cross Reference -----------------------------
#define SFF8436_MEDIA_TYPE_OFFSET                147
#define SFF8436_LINK_LEN_OFFSET                  146
#define SFF8436_VENDOR_OFFSET                    148
#define SFF8436_EXT_IDENTIFIER_OFFSET            129
#define SFF8436_POWER_DATA_OFFSET                93
#define SFF8436_LOW_POWER_MASK                   0x02

// SFF-8024 TABLE 4-1 IDENTIFIER VALUES
#define SFF8024_TYPE_UNKNOWN                     0x00
#define SFF8024_TYPE_GBIC                        0x01
#define SFF8024_TYPE_MOTHERBOARD                 0x02
#define SFF8024_TYPE_SFP                         0x03
#define SFF8024_TYPE_300P_XBI                    0x04
#define SFF8024_TYPE_XENPAK                      0x05
#define SFF8024_TYPE_XFP                         0x06
#define SFF8024_TYPE_XFF                         0x07
#define SFF8024_TYPE_XFP_E                       0x08
#define SFF8024_TYPE_XPAK                        0x09
#define SFF8024_TYPE_X2                          0x0A
#define SFF8024_TYPE_DWDM_SFP                    0x0B
#define SFF8024_TYPE_QSFP_INF8438                0x0C
#define SFF8024_TYPE_QSFP_PLUS                   0x0D
#define SFF8024_TYPE_CXP                         0x0E
#define SFF8024_TYPE_SMMHD4X                     0x0F
#define SFF8024_TYPE_SMMHD8X                     0x10
#define SFF8024_TYPE_QSFP28                      0x11
#define SFF8024_TYPE_CXP2                        0x12
#define SFF8024_TYPE_CDFP                        0x13
#define SFF8024_TYPE_SMMHD4X_FANOUT              0x14
#define SFF8024_TYPE_SMMHD8X_FANOUT              0x15
#define SFF8024_TYPE_CDFP_3                      0x16
#define SFF8024_TYPE_uQSFP                       0x17
#define SFF8024_TYPE_QSFPDD                      0x18
#define SFF8024_TYPE_OSFP8X                      0x19
#define SFF8024_TYPE_SFPDD                       0x1A
#define SFF8024_TYPE_DSFP                        0x1B
#define SFF8024_TYPE_MINILINK4X                  0x1C
#define SFF8024_TYPE_MINILINK8X                  0x1D
#define SFF8024_TYPE_QSFP_PLUS_CMIS              0x1E

// SFF-8636 Byte @ 93
#define SFF8636_SW_RESET                         0x80
#define SFF8636_PWR_CLASS_8_ENABLE               0x08
#define SFF8636_PWR_CLASS_567_ENABLE             0x04
#define SFF8636_PWR_CLASS_LOW                    0x02
#define SFF8636_PWR_OVERRIDE                     0x01

// SFF-8636 Extended Identifier Byte @ Page 00h byte 129
#define SFF8636_PWR_CLASS_1234_MASK              0xC0
#define SFF8636_PWR_CLASS_1                      0x00
#define SFF8636_PWR_CLASS_2                      0x40
#define SFF8636_PWR_CLASS_3                      0x80
#define SFF8636_PWR_CLASS_4_AND_UP               0xC0
#define SFF8636_PWR_CLASS_8                      0x20
#define SFF8636_CLEI_IN_PG02                     0x10
#define SFF8636_CDR_IN_TX                        0x08
#define SFF8636_CDR_IN_RX                        0x04
#define SFF8636_PWR_CLASS_567_MASK               0x03
#define SFF8636_SEE_PWR_CLASS_1234               0x00
#define SFF8636_PWR_CLASS_5                      0x01
#define SFF8636_PWR_CLASS_6                      0x02
#define SFF8636_PWR_CLASS_7                      0x03

// Common Management Interface Specification (CMIS) ---------------------------
#define CMIS_MEDIA_TYPE_OFFSET                   212
#define CMIS_CABLE_LEN_OFFSET                    202
#define CMIS_VENDOR_OFFSET                       129

// CMIS and SFF-8636 Media/Transmitter Technology Types
#define MEDIA_850_NM_VCSEL                       0x00
#define MEDIA_1310_NM_VCSEL                      0x01
#define MEDIA_1550_NM_VCSEL                      0x02
#define MEDIA_1310_NM_FP                         0x03
#define MEDIA_1310_NM_DFB                        0x04
#define MEDIA_1550_NM_DFB                        0x05
#define MEDIA_1310_NM_EML                        0x06
#define MEDIA_1550_NM_EML                        0x07
#define MEDIA_OTHERS                             0x08
#define MEDIA_1490_NM_DFB                        0x09
#define MEDIA_COPPER_UNEQ                        0x0A
#define MEDIA_COPPER_PASSIVE_EQ                  0x0B
#define MEDIA_COPPER_NEAR_FAR_LMT_ACT_EQ         0x0C
#define MEDIA_COPPER_FAR_LMT_ACT_EQ              0x0D
#define MEDIA_COPPER_NEAR_LMT_ACT_EQ             0x0E
#define MEDIA_COPPER_LIN_ACT_EQ                  0x0F

/**
 * enum qsfp_format - cable spec types
 *
 * @QDD_FMT_UNKNOWN:       Format unknown
 * @QDD_SFF8636:           Format compatible with SFF-8636
 * @QDD_CMIS:              Format compatible with CMIS
 *
 * QSFP / QSFP-DD Protocol/Specification Types
 */
enum qsfp_format {
	QDD_FMT_UNKNOWN = 0,
	QDD_SFF8636,
	QDD_CMIS,
};

/**
 * enum cass_headshell_status - cable status
 *
 * @CASS_HEADSHELL_STATUS_UNKNOWN:       State unknown
 * @CASS_HEADSHELL_STATUS_PRESENT:       A headshell is inserted
 * @CASS_HEADSHELL_STATUS_NOT_PRESENT:   A headshell is not inserted
 * @CASS_HEADSHELL_STATUS_ERROR:         A headshell has error
 * @CASS_HEADSHELL_STATUS_NO_DEVICE:     No headshell hardware device
 *
 * The status of the media headshell
 */
enum cass_headshell_status {
	CASS_HEADSHELL_STATUS_UNKNOWN          = 0,
	CASS_HEADSHELL_STATUS_PRESENT          = 1<<0,
	CASS_HEADSHELL_STATUS_NOT_PRESENT      = 1<<1,
	CASS_HEADSHELL_STATUS_ERROR            = 1<<2,
	CASS_HEADSHELL_STATUS_NO_DEVICE        = 1<<3,
};

int cass_is_cable_present(struct cass_dev *hw);
int cass_headshell_power_up(struct cass_dev *hw, u8 qsfp_format);
int cass_parse_heashell_data(struct cass_dev *hw, struct sbl_media_attr *attr,
			     u8 *qsfp_format);
int cass_link_headshell_insert(struct cass_dev *hw,
			       struct sbl_media_attr *mattr);
int cass_link_headshell_remove(struct cass_dev *hw);
int cass_link_headshell_error(struct cass_dev *hw);
int cass_link_media_config(struct cass_dev *hw, struct sbl_media_attr *attr);
const char *cass_link_headshell_state_str(enum cass_headshell_status state);

#endif	/* _CASS_CABLE_H */
