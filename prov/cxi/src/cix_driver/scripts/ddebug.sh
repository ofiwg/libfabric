#!/bin/bash

# Manage module dynamic_debug
# Write to /sys/kernel/debug/dynamic_debug/control to control what
# pr_debug() prints will be enabled to print to the console.
# Changes the printk level 8.

flags="pt"
MODULE=cxi_ss1

echo 8 > /proc/sys/kernel/printk

while getopts "dDf:F:hl:omM:" OPTION; do
        case $OPTION in
        h)      # help
                echo "${0##*/} [-M module] [-F flags] [-hdDom] [-f func1,func2] [-l file,line]"
                echo "          Manage dynamic_debug for a module (default cxi_ss1)"
                echo "          -h This help message"
                echo "          -M Turn on debug for an alternate module (list first)"
                echo "          -F Alternate flags (flmpt) Default pt"
                echo "          -d Show current dynamic_debug entries"
                echo "          -D Set more_debug module parameter"
                echo "          -f <func1,func2,...> Function list - comma separated functions to probe"
                echo "          -l <file,lineno> File and line number to probe"
                echo "          -o Turn off debug"
                echo "          -m Turn on all debug for module"
                ;;
        d)      # dump
                cat /sys/kernel/debug/dynamic_debug/control | grep $MODULE | grep =$flags
                ;;
        f)      # func
                functions="$OPTARG"
                echo functions $functions
                for func in ${functions//,/ }
                do
                        echo func $func
                        echo "echo -n \"func $func +$flags\" > /sys/kernel/debug/dynamic_debug/control"
                        echo -n "func $func +$flags" > /sys/kernel/debug/dynamic_debug/control
                done
                ;;
        D)      # more atu debug
                echo add more debug
                echo 1 > /sys/module/$MODULE/parameters/more_debug
                ;;
        o)      # turn off driver debug
                echo turn off driver debug
                echo -n "module $MODULE -$flags" > /sys/kernel/debug/dynamic_debug/control

                echo 0 > /sys/module/$MODULE/parameters/more_debug
                echo 7 > /proc/sys/kernel/printk
                ;;
        m)      # default full cxi_ss1 module
                echo enabling all of $MODULE pr_debug
                echo "echo -n \"module $MODULE +$flags\" > /sys/kernel/debug/dynamic_debug/control"
                echo -n "module $MODULE +$flags" > /sys/kernel/debug/dynamic_debug/control
                ;;
        l)      # file and line number
                echo file, line number ${OPTARG}
                ARRAY=(${OPTARG//,/ })
                echo -n "file ${ARRAY[0]} line ${ARRAY[1]} +$flags" > /sys/kernel/debug/dynamic_debug/control
                ;;
        M)      # Alternate module
                MODULE="$OPTARG"
                ;;
        F)      # Alternate flags
                flags="$OPTARG"
                ;;
        esac
done
