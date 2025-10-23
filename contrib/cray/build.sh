#!/usr/bin/env bash
#
# Copyright 2024 Hewlett Packard Enterprise Development LP. All rights reserved.
#

set -Exeuo pipefail

CE_BUILD_SCRIPT_REPO=hpc-shs-ce-devops

# Use appropriate ce-devops branch
if [[ -n "${BRANCH_NAME:-}" && "$BRANCH_NAME" =~ "release/shs-" ]]; then
    CE_CONFIG_BRANCH=$BRANCH_NAME
elif [[ -n "${CHANGE_TARGET:-}" && "$CHANGE_TARGET" =~ "release/shs-" ]]; then
    CE_CONFIG_BRANCH=$CHANGE_TARGET
else
    CE_CONFIG_BRANCH=${CE_CONFIG_BRANCH:-main}
fi

if [ -d "${CE_BUILD_SCRIPT_REPO}" ]; then
    git -C ${CE_BUILD_SCRIPT_REPO} fetch
    git -C ${CE_BUILD_SCRIPT_REPO} checkout ${CE_CONFIG_BRANCH}
    git -C ${CE_BUILD_SCRIPT_REPO} pull
else
    git clone --branch "${CE_CONFIG_BRANCH}" https://$HPE_GITHUB_TOKEN@github.hpe.com/hpe/${CE_BUILD_SCRIPT_REPO}.git
fi

. "${CE_BUILD_SCRIPT_REPO}/build/libfabric/build.sh"
