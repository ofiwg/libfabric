#!/bin/bash

PROJECT="NETC"
TARGET_ARCH="x86_64"
DEV_NAME="dev"
BRANCH_NAME="master"
IYUM_REPO_NAME_1="os-networking-team"

if [[ "${PRODUCT}" = "" ]]
then
    PRODUCT="shasta-premium"
fi

if [[ "${TARGET_OS}" = "" ]]
then
    TARGET_OS="sle15_cn"
fi

echo "$0: --> PRODUCT: '${PRODUCT}'"
echo "$0: --> TARGET_OS: '${TARGET_OS}'"

ZYPPER_OPTS="--verbose --non-interactive"
RPMS="cray-libcxi-devel"

URL_PREFIX="http://car.dev.cray.com/artifactory"
URL_SUFFIX="${TARGET_OS}/${TARGET_ARCH}/${DEV_NAME}/${BRANCH_NAME}"

# URL="http://car.dev.cray.com/artifactory/"
# URL+="${PRODUCT}/${PROJECT}/${TARGET_OS}/${TARGET_ARCH}/"
# URL+="${DEV_NAME}/${BRANCH_NAME}/"
URL="${URL_PREFIX}/${PRODUCT}/${PROJECT}/${URL_SUFFIX}"

URL_INT="${URL_PREFIX}/internal/${PROJECT}/${URL_SUFFIX}"

# URL_SSHOT="http://car.dev.cray.com/artifactory/"
# URL_SSHOT+="${PRODUCT}/SSHOT/${TARGET_OS}/${TARGET_ARCH}/"
# URL_SSHOT+="${DEV_NAME}/${BRANCH_NAME}/"
URL_SSHOT="${URL_PREFIX}/${PRODUCT}/SSHOT/${URL_SUFFIX}"

URL_SSHOT_INT="${URL_PREFIX}/internal/SSHOT/${TARGET_OS}/${TARGET_ARCH}/"
URL_SSHOT_INT+="predev/integration/"

if command -v yum > /dev/null; then
    yum-config-manager --add-repo=$URL
    yum-config-manager --add-repo=$URL_SSHOT

    yum-config-manager --setopt=gpgcheck=0 --save

    yum install -y $RPMS
elif command -v zypper > /dev/null; then
    zypper $ZYPPER_OPTS addrepo --no-gpgcheck --check --priority 1 \
    	--name=$IYUM_REPO_NAME_1 $URL $IYUM_REPO_NAME_1
    zypper $ZYPPER_OPTS addrepo --no-gpgcheck --check --priority 1 \
    	--name=${IYUM_REPO_NAME_1}_SSHOT $URL_SSHOT \
        ${IYUM_REPO_NAME_1}_SSHOT

    if [[ $TARGET_OS =~ ncn$ ]]; then
        zypper $ZYPPER_OPTS addrepo --no-gpgcheck --check --priority 10 \
            --name=${IYUM_REPO_NAME_1}_internal $URL_INT \
            ${IYUM_REPO_NAME_1}_internal
        zypper $ZYPPER_OPTS addrepo --no-gpgcheck --check --priority 10 \
            --name=${IYUM_REPO_NAME_1}_SSHOT_internal $URL_SSHOT_INT \
            ${IYUM_REPO_NAME_1}_SSHOT_internal
    fi

    zypper refresh
    zypper $ZYPPER_OPTS install $RPMS
else
    "Unsupported package manager or package manager not found -- installing nothing"
fi
