#!/bin/sh

cat onepage.list | xargs cat | pandoc -o onepage.md
echo '---' > temp
echo 'layout: page' >> temp
echo 'title: All-in-one Man Page' >> temp
echo "tagline: Libfabric Programmer's Manual" >> temp
echo '---' >> temp
echo '' >> temp
cat onepage.md >> temp && mv temp onepage.md

