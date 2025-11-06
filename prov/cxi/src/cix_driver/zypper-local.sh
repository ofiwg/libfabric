WORKSPACE="WORKSPACE"

ZYPPER_ROOT=`pwd`/${WORKSPACE}/zypp

ZYPPER_OPTS=$(echo "
-D $ZYPPER_ROOT/etc/zypp/repos.d
-C $ZYPPER_ROOT/var/cache/zypp
--raw-cache-dir $ZYPPER_ROOT/var/cache/zypp/raw
--solv-cache-dir $ZYPPER_ROOT/var/cache/zypp/solv
--pkg-cache-dir $ZYPPER_ROOT/var/cache/zypp/packages
-vv
--non-interactive
--no-gpg-checks
")

if command -v zypper > /dev/null; then

    if [ $UID -eq 0 ]; then
	ZYPPER_COMMAND="zypper $ZYPPER_OPTS"
    else
	ZYPPER_COMMAND="sudo zypper $ZYPPER_OPTS"
    fi
else
    ZYPPER_COMMAND=""
fi
