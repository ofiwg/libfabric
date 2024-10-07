#!/bin/bash

# Cache head commit which will be cherry-picked later
head_commit=$(git rev-parse HEAD)

git checkout -b rebase-test-branch
db=$(git remote show https://github.com/ofiwg/libfabric.git | grep 'HEAD branch' | cut -d' ' -f5)
mb=$(git merge-base origin/${db} HEAD)

# Run a shorten test suite against each commits except the head commit.
git reset --hard HEAD~1
git rebase ${mb} --exec "bash ./cxi_vm_commit.sh -s"
if [[ $? -ne 0 ]]; then
    exit 1
fi

# Run longer test suite against all commits together.
git cherry-pick ${head_commit}
bash ./cxi_vm_commit.sh
