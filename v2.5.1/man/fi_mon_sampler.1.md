---
layout: page
title: fi_mon_sampler(1)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}


# NAME

fi_mon_sampler  \- Simple sampler for ofi_hook_monitor provider.


# SYNOPSIS
```
 fi_mon_sampler [OPTIONS] <target>		sample from file(s) at <target>
```

# DESCRIPTION

Extract data from the ofi_hook_monitor provider via communication files. `<target>` can either be
one communication file or a folder of files. Data is exported based on `-f <format>` and either printed
to stdout (only for single files), or stored per communication file at `-o <outpath>`.
The sampler can watch the communication files for changes via the option `-w <msec>` 
for repeated sampling.

The name format of the output files is based on the ofi_hook_monitor provider and is as follows:
`<ppid>_<pid>_<sequential id>_<job id>_<provider name>`.
`ppid` and `pid` are taken from the perspective of the monitored application.
In a batched environment running SLURM, `job id` is set to the SLURM job ID, otherwise it is set to 0.


# HOW TO RUN

Launch a libfabric application with `FI_HOOK=monitor` to enable the ofi_hook_monitor provider. 
Adjust the monitor provider settings according to [`fi_hook`(7)](fi_hook.7.html).

Then launch the sampler via `fi_mon_sampler -o <output> <target>`. 
By default, the ofi_hook_monitor provider stores data at `/dev/shm/ofi/<uid>/<hostname>`.

The sampler will generate output files in the directory specified at `<output>`, one for each monitored provider.

# OPTIONS

*-w \<msec\>*
: Watch files for changes, check every \<msec\> milliseconds.

*-f \<format\>*
: Output format. Currently only supports CSV.

*-o \<outpath\>*
: Output file path. Uses stdout if unset.


# USAGE EXAMPLES

Launch a libfabric application and enable the ofi_hook_monitor provider:
```bash
FI_HOOK=monitor fi_pingpong [OPTIONS]
```
Launch another `fi_pingpong` with the respective settings.

Finally, launch the sampler:
```bash
fi_mon_sampler -o $HOME -w 1000 -f csv /dev/shm/ofi/$UID/$HOSTNAME
```


# OUTPUT

Output files will be generated in the folder specified at `-o <output>`.

In `-f csv` mode, this will contain a CSV file with data for all monitored libfabric functions.
For each function, both the `count` and `sum` counters are exported, 
indicated by the column name suffix `_c` and `_s` respectively.
In addition, each function is monitored for each data size bucket.
Refer to [`fi_hook`(7)](fi_hook.7.html) for more details.

Example CSV output, first four columns, first three rows:

```csv
mon_recv_0_64_c,mon_recv_0_64_s,mon_recv_64_512_c,mon_recv_64_512_s
0,0,0,0
22529,0,0,0
113664,0,0,0
```

# SEE ALSO

[`fi_hook`(7)](fi_hook.7.html)
