#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
# Copyright 2020 Hewlett Packard Enterprise Development LP

git checkout -b rebase-test-branch
./autogen.sh
./configure --with-criterion=$(realpath ../Criterion/build/install)

rm -f commit-results.tap
git rebase `git merge-base origin/main HEAD` --exec 'set -e; git log -1; SHORT=$(git rev-parse --short HEAD); TESTS="not ok"; COMMIT="not ok"; make clean && make dist && make -j 8 && ssh -tt localhost "cd ~/workspace/workspace/os-networking-team/cassini-vm/PR-libcxi/ && make check"; ECODE=1; grep -q "not ok" tests/libcxi_test.tap || ECODE=$(($?^1)); [[ "$ECODE" -eq "0" ]] && TESTS="ok"; echo "${TESTS} - Commit ${SHORT} Tests Status" >> commit-results.tap; git log -1 | grep -q "Signed-off-by: "; [[ "$?" -eq "0" ]] && COMMIT="ok"; echo "${COMMIT} - Commit ${SHORT} Signed-off" >> commit-results.tap; exit $ECODE'

sed -i "1s/^/1..$(wc -l commit-results.tap | cut -c1)\n/" commit-results.tap
sed -i '1s/^/TAP version 13\n/' commit-results.tap
mv -f commit-results.tap tests/
