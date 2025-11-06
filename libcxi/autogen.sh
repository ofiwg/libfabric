#! /bin/sh
# SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
# Copyright 2018 Hewlett Packard Enterprise Development LP

set -x

if [ ! -f .git/hooks/pre-commit ] ; then
	contrib/install-git-hook.sh
	echo "Installed pre-commit hook."
fi

autoreconf -ivf
