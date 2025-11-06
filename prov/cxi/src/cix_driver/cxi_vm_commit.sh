#!/bin/bash
make -j8
# We need a TTY, otherswise the tests will not finish. So ssh to self.
ssh -tt localhost "cd ~/workspace/workspace/os-networking-team/cassini-vm/cxi-driver/; make check"
