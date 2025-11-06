#!/bin/bash
git checkout -b rebase-test-branch

# Build only first as it's fast
git rebase `git merge-base origin/main HEAD` --exec 'set -e; git log -1 && make clean && make -j8'

# Then build and test
# We need a TTY, otherswise the tests will not finish. So ssh to self.
git rebase `git merge-base origin/main HEAD` --exec 'set -e; git log -1 && make clean && make -j8 && ssh -tt localhost "cd ~/workspace/workspace/os-networking-team/cassini-vm/PR-cxi-driver/; make check"'
