#!/bin/sh

cat onepage.list | xargs pandoc -o onepage.md
cat onepage-head.txt onepage.md > temp && mv temp onepage.md

