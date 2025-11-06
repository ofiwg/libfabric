---
title: CXI_SERVICE(1) Version 0.4.0 | CXI Services
date: 2022-07-19
---

# NAME

cxi_service - CXI Service Utility


# SYNOPSIS

**cxi_service** __COMMAND__ [*OPTIONS*]


# DESCRIPTION

**cxi_service** is a utility which allows for displaying and modifying (some)
CXI Services.

# COMMANDS

**create**
: Create a service based on the yaml file specified by the -y flag.

**delete**
: Delete the service specified by the -s flag. The DEFAULT Service cannot be deleted

**disable**
: Disable (but not delete) the service specified by the -s flag.

**enable**
: Enable the service specified by the -s flag.

**list**
: Output a list of all CXI devices

# OPTIONS

**-d, \-\-device**=*DEV*
: The Cassini NIC device name to report. When unspecified, cxi0 is chosen
by default.

**-h, \-\-help**
: Display the help text and exit.

**-v, \-\-verbose**
: Increase verbosity of output messages.

**-s, \-\-svc-id**
: Apply commands to only the svc_id specified.

**-V, \-\-version**
: Display the program version and exit.

**-y, \-\-yaml-file**
: Path to yaml file to use with the 'create' command (see FILES section for example).


# EXAMPLES
```
$ cxi_service -h
cxi_service - CXI Service Utility

Usage: cxi_service <COMMAND> [options]
 -d --device=DEV       CXI device. Default is cxi0
 -h --help             Show this help
 -s --svc_id           Only apply commands to this specific svc_id
 -V --version          Print the version and exit
 -v --verbose          Increase verbosity
 -y --yaml_file        Path to yaml file to use with 'create' command

Commands:
 create                Create a service
 delete                Delete the service specified by the -s flag
 disable               Disable the service specified by the -s flag
 enable                Enable the service specified by the -s flag
 list                  List all services for a device

$ cxi_service -V
cxi_service version: 0.5.0

$ cxi_service list -d cxi1
cxi1
----
 Total Device Resources
 ----------------------
 ACs:  1022
 CTs:  2047
 EQs:  2047
 LEs:  16384
 PTEs: 2048
 TGQs: 512
 TXQs: 1024
 TLEs: 2048
 --------------------------
 ID: 1 (DEFAULT)
   Enabled            : Yes
   System Service     : No
   Restricted Members : No
   Restricted VNIs    : No
   Restricted TCs     : No
   Resource Limits    : No


$ cxi_service list -v
cxi0
----
 Total Device Resources
 ----------------------
 ACs:  1022
 CTs:  2047
 EQs:  2047
 LEs:  16384
 PTEs: 2048
 TGQs: 512
 TXQs: 1024
 TLEs: 2048
 --------------------------
 ID: 1 (DEFAULT)
   LNIs/RGID          : 1
   Enabled            : Yes
   System Service     : No
   Restricted Members : No
   ---> Valid Members : All uids/gids
   VNIs               : 1 10
   Restricted TCs     : No
   ---> Valid TCs     : All
   Resource Limits    : Yes
          ---------------------------------
          |  Max    |  Reserved |  In Use |
          ---------------------------------
     ACs  |  1022   |   0       |  0      |
     CTs  |  2047   |   0       |  0      |
     EQs  |  2047   |   0       |  0      |
     LEs  |  16384  |   0       |  0      |
     PTEs |  2048   |   0       |  0      |
     TGQs |  512    |   0       |  0      |
     TXQs |  1024   |   0       |  0      |
     TLEs |  512    |   512     |  0      |
          ---------------------------------
 ------------------------------------------
 ID: 2
   LNIs/RGID          : 1
   Enabled            : Yes
   System Service     : No
   Restricted Members : Yes
   ---> Valid Members : uid=1 gid=2
   VNIs               : 64-127
   Exclusive CP       : Yes
   Restricted TCs     : Yes
   ---> Valid TCs     : DEDICATED_ACCESS LOW_LATENCY BULK_DATA BEST_EFFORT
   Resource Limits    : Yes
          ---------------------------------
          |  Max    |  Reserved |  In Use |
          ---------------------------------
     ACs  |  1      |   1       |  0      |
     CTs  |  1      |   1       |  0      |
     EQs  |  1      |   1       |  0      |
     LEs  |  1      |   1       |  0      |
     PTEs |  1      |   1       |  0      |
     TGQs |  1      |   1       |  0      |
     TXQs |  1      |   1       |  0      |
     TLEs |  9      |   9       |  0      |
          ---------------------------------

$ cxi_service list --svc_id 1
 --------------------------
 ID: 1 (DEFAULT)
   LNIs/RGID          : 1
   Enabled            : Yes
   System Service     : No
   Restricted Members : No
   Restricted TCs     : No
   Resource Limits    : Yes


$ cxi_service delete -s 1
cxi_service: Default service cannot be deleted.

$ cxi_service delete -s 2
Successfully deleted service: 2

$ cxi_service create -y $PATH_TO_YAML_FILE

# FILES

 _share/cxi/cxi_service_template.yaml_
 ```
 Sample yaml file to be used with the "create" command
 ```

# SEE ALSO

 **cxi_service**(7)
