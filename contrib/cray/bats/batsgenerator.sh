#!/bin/bash                                                                                                                              
# Generate a bats file to run Intel MPI Benchmarks
# Assumes IMB test suite has been installed and included in Jenkinsfile.verbs file
# Example:
# Add IMB-EXT Windows test running 20 ranks, 5 ranks per node to imb.bats
#       ./batsgenerator.sh IMB-EXT windows 20 5 imb.bats

# Insert shebang and load test helper
shebang="#!/usr/bin/env bats\n\n"

# Command line input: test suite
# E.g. IMB-EXT
test_suite=$1
shift

# Command line input: benchmark
# E.g. windows
benchmark=$1
shift

# Command line input: number of ranks
# E.g. 20
num_ranks=$1
shift

# Command line input: number of ranks per node (rpn)
# E.g. 5
num_rpn=$1
shift

#Command line input: name of bats file
# E.g. imb.bats
bats_file=$1

# Add test to .bats file
if [ -f "${bats_file}" ]; then
        sed -e "s/@TEST_SUITE@/${test_suite}/g" \
        -e "s/@BENCHMARK@/${benchmark}/g" \
        -e "s/@RANKS@/${num_ranks}/g" \
        -e "s/@RPN@/${num_rpn}/g" \
        benchmark.template >> ${bats_file}
else
        printf "${shebang}load test_helper\n\n" >> ${bats_file}
        sed -e "s/@TEST_SUITE@/${test_suite}/g" \
        -e "s/@BENCHMARK@/${benchmark}/g" \
        -e "s/@RANKS@/${num_ranks}/g" \
        -e "s/@RPN@/${num_rpn}/g" \
        benchmark.template >> ${bats_file}
fi
