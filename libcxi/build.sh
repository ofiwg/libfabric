#!/usr/bin/env bash
#
# Copyright 2024 Hewlett Packard Enterprise Development LP. All rights reserved.
#

set -Eeuo pipefail

CE_BUILD_SCRIPT_REPO=hpc-shs-ce-devops
CE_CONFIG_BRANCH=${CE_CONFIG_BRANCH:-main}

if [ -d ${CE_BUILD_SCRIPT_REPO} ]; then
    git -C ${CE_BUILD_SCRIPT_REPO} fetch 
    git -C ${CE_BUILD_SCRIPT_REPO} checkout ${CE_CONFIG_BRANCH}
    git -C ${CE_BUILD_SCRIPT_REPO} pull
else
    git clone --branch "${CE_CONFIG_BRANCH}" https://$HPE_GITHUB_TOKEN@github.hpe.com/hpe/${CE_BUILD_SCRIPT_REPO}.git
fi

. ${CE_BUILD_SCRIPT_REPO}/build/sh/rpmbuild/load.sh

setup_dst_env

dst_build_rpm -c ${CE_BUILD_SCRIPT_REPO}/build/configs/libcxi.yaml $@
dst_build_deb --yamlfile ${CE_BUILD_SCRIPT_REPO}/build/configs/${CE_CONFIG_FILE} --ps ${PRODUCT} --main-branch "main" $@

