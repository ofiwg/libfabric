#!/bin/bash
git checkout -b rebase-test-branch
mb=$(git merge-base origin/v1.15.x-ss11 HEAD)
git rebase ${mb} --exec 'bash ./cxi_vm_commit.sh'
