#!/bin/bash

set -e

function check_executable {
    if [[ ! -x $1 ]] ; then
        echo could not find executable: $1
        exit 1
    fi
}

function check_directory {
    if [[ ! -d $1 ]] ; then
        echo could not find directory: $1
        exit 1
    fi
}

function check_environment {
    var=$1
    val=$(eval "echo \$$var")
    if [[ -z "$val" ]] ; then
        echo could not find environment variable: $var
        exit 1
    fi
}

function find_executable {
    if [[ -z "$(which $1)" ]] ; then
        echo could not find executable: $1
        exit 1
    fi
}

# check prerequisite environment variables
check_environment ROOT_BUILD_PATH
check_environment FABTEST_PATH
check_environment LIBFABRIC_BUILD_PATH
check_environment OMB_BUILD_PATH
check_environment MPICH_PATH

# check directories
check_directory $ROOT_BUILD_PATH
check_directory $FABTEST_PATH
check_directory $OMB_BUILD_PATH
check_directory $MPICH_PATH
check_directory $SFT_INSTALL_PATH

# check prerequisite installed software packages
find_executable srun
check_executable $FABTEST_PATH/bin/runfabtests.sh
check_executable $OMB_BUILD_PATH/pt2pt/osu_bw
check_executable $SFT_INSTALL_PATH/bin/ci-all.sh

exit 0
