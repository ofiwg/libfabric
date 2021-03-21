#!/bin/bash

check() {
    a=$*

    echo checking value: $a
    if test -n "`echo $a | grep -i secret`"; then
        echo "yow: it has the word secret!"
    fi
    if test -n "`echo $a | grep -i jsquyres`"; then
        echo "yow: it has the word jsquyres!"
    fi
    if test -n "`echo $a | grep -i ecc`"; then
        echo "yow: it has the word ecc!"
    fi
}

echo yow: this is env val: $VAL
check "$VAL"

arg=$1
echo yow: this is arg: $arg
check "$arg"
