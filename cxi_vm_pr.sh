#!/bin/bash
git checkout -b rebase-test-branch
db=$(git remote show git@github.hpe.com:hpe/hpc-shs-libfabric-netc.git | grep 'HEAD branch' | cut -d' ' -f5)
mb=$(git merge-base origin/${db} HEAD)
git rebase ${mb} --exec 'bash ./cxi_vm_commit.sh'
