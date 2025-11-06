#!/bin/sh -x

set -ex

. $ROOT_DIR/vars.sh
. $ROOT_DIR/zypper-local.sh

TESTDIR=$ROOT_DIR/tests/dkms
DKMSTEST_SSH_USER=dkmstest

TEST_ARCH=$(grep $TARGET_OS $TESTDIR/os-machine-mapping | cut -d: -f1)
TEST_MACHINE=$(grep $TARGET_OS $TESTDIR/os-machine-mapping | cut -d: -f2)
TEST_IMAGE=$(grep $TARGET_OS $TESTDIR/os-machine-mapping | cut -d: -f3)

if [ $TEST_ARCH ] ; then
    echo "Test arch: \"$TEST_ARCH\""
else
    echo "Unable to find test arch for os \"$TARGET_OS\""
    exit 0
fi

if [ $TEST_MACHINE ] ; then
    echo "Test machine: \"$TEST_MACHINE\""
else
    echo "Unable to find test machine for os \"$TARGET_OS\""
    exit 0
fi

if [ $TEST_IMAGE ] ; then
    echo "Test image: \"$TEST_IMAGE\""
else
    echo "Unable to find test image for os \"$TARGET_OS\""
    exit 0
fi

cat >> $ROOT_DIR/vars.sh <<- END
export DKMSTEST_SSH_USER=dkmstest
export TEST_ARCH=${TEST_ARCH}
export TEST_MACHINE=${TEST_MACHINE}
export TEST_IMAGE=${TEST_IMAGE}
END

. $ROOT_DIR/vars.sh

echo "$0: --> ROOT_DIR:	'${ROOT_DIR}'"
echo "$0: --> TEST_ARCH: '${TEST_ARCH}'"
echo "$0: --> TEST_MACHINE: '${TEST_MACHINE}'"
echo "$0: --> TEST_IMAGE: '${TEST_IMAGE}'"

createrepo RPMS

# Package up the files we need

if [ -f cxi-driver-dkms.tar.gz ]; then
    rm cxi-driver-dkms.tar.gz
fi

export DKMS_TEMPDIR=$(ssh $TEST_MACHINE mktemp -d)
echo "$0: --> DKMS_TEMPDIR: '${DKMS_TEMPDIR}'"
echo "export DKMS_TEMPDIR=${DKMS_TEMPDIR}" >> vars.sh

cat > $ROOT_DIR/start-test-container.sh <<- END
#!/bin/sh -x

. $DKMS_TEMPDIR/vars.sh

docker run --security-opt apparmor=unconfined --net=host -u $DKMSTEST_SSH_USER --mount src=$DKMS_TEMPDIR,target=/home/dkmstest,type=bind --mount src=/tmp,target=/tmp,type=bind --mount src=/root,target=/root,type=bind --rm --env-file $DKMS_TEMPDIR/env.sh --entrypoint $DKMS_TEMPDIR/run-test.sh $TEST_IMAGE
END

chmod a+x start-test-container.sh

cat $ROOT_DIR/vars.sh | cut -d " " -f 2 > $ROOT_DIR/env.sh

tar -C .. -zcvf cxi-driver-dkms.tar.gz hpc-shs-cxi-driver/WORKSPACE/RPMS hpc-shs-cxi-driver/WORKSPACE/SRPMS hpc-shs-cxi-driver/RPMS  hpc-shs-cxi-driver/hpc-sshot-slingshot-version hpc-shs-cxi-driver/tests/dkms hpc-shs-cxi-driver/env.sh hpc-shs-cxi-driver/vars.sh hpc-shs-cxi-driver/zypper-local.sh 

scp cxi-driver-dkms.tar.gz $TEST_MACHINE:$DKMS_TEMPDIR
scp start-test-container.sh $TEST_MACHINE:$DKMS_TEMPDIR
scp vars.sh $TEST_MACHINE:$DKMS_TEMPDIR
scp vars.sh $TEST_MACHINE:$DKMS_TEMPDIR/.bashrc
scp env.sh $TEST_MACHINE:$DKMS_TEMPDIR
scp zypper-local.sh $TEST_MACHINE:$DKMS_TEMPDIR
scp tests/dkms/run-test.sh $TEST_MACHINE:$DKMS_TEMPDIR

rm $ROOT_DIR/env.sh

ssh $TEST_MACHINE chown -R $DKMSTEST_SSH_USER.users $DKMS_TEMPDIR
ssh $TEST_MACHINE chmod a+rwx $DKMS_TEMPDIR
ssh $TEST_MACHINE chmod -R a+rw $DKMS_TEMPDIR
ssh $TEST_MACHINE chmod -R a+rwx $DKMS_TEMPDIR/*.sh
