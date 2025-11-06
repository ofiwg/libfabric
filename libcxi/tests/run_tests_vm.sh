#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
# Copyright 2020 Hewlett Packard Enterprise Development LP

cd $(dirname $0)
. ./virtualize.sh

./libcxi_test --verbose --tap=libcxi_test.tap --tap=- -j1
