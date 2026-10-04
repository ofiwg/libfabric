#!/usr/bin/env bash
#
# Copyright 2024 Hewlett Packard Enterprise Development LP. All rights reserved.
#

set -Eeuox pipefail

CE_BUILD_SCRIPT_REPO=hpc-shs-ce-devops

if [ -d ${CE_BUILD_SCRIPT_REPO} ]; then
    git -C ${CE_BUILD_SCRIPT_REPO} fetch
    git -C ${CE_BUILD_SCRIPT_REPO} checkout "main"
    git -C ${CE_BUILD_SCRIPT_REPO} pull
else
    git clone --branch "main" https://$HPE_GITHUB_TOKEN@github.hpe.com/hpe/${CE_BUILD_SCRIPT_REPO}.git
fi

. "${CE_BUILD_SCRIPT_REPO}/build/sh/rpmbuild/build-common.sh"
