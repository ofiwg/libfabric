# Copyright 2021,2025 Hewlett Packard Enterprise Development LP
%define release_extra 0

%{!?dkms_source_tree:%define dkms_source_tree /usr/src}
%define dkms_source_dir %{dkms_source_tree}/%{name}-%{version}-%{release}

# Enable AMD GPU dependencies by building with --with=amdgpu rpmbuild option
%bcond_with amdgpu

# Enable Nvidia GPU dependencies by building with --with=nvidiagpu rpmbuild option
%bcond_with nvidiagpu

%if 0%{?rhel}
%define distro_kernel_package_name kmod-%{name}
%else
%define distro_kernel_package_name %{name}-kmp
%endif

# Exclude -preempt kernel flavor, this seems to get built alongside the -default
# flavor for stock SLES. It doesn't get used, and its presence can cause issues
%define kmp_args_common -x azure preempt -p %{_sourcedir}/%name.rpm_preamble

%if 0%{?rhel}
# On RHEL, override the kmod RPM name to include the kernel version it was built
# for; this allows us to package the same driver for multiple kernel versions.
%define kmp_args -n %name-k%%kernel_version %kmp_args_common
%else
%define kmp_args %kmp_args_common
%endif

Name:           cray-cxi-driver
Version:        1.0.0
Release:        %(echo ${BUILD_METADATA})
Summary:        HPE Cassini Driver
License:        GPL-2.0
Source0:        %{name}-%{version}.tar.gz
Prefix:         /usr

BuildRequires:  cray-cassini-headers-user kernel-devel
BuildRequires:  cray-slingshot-base-link-devel
BuildRequires:  cassini2-firmware-devel
BuildRequires:  sl-driver-devel

%if %{with nvidiagpu}
%if %{with shasta_premium}
# COS needs the nvidia-gpu-build helper package to install cuda-drivers
# successfully
%if "%(echo $SHS_NEW_BUILD_SYSTEM)" == "y"
%else
BuildRequires: nvidia-gpu-build
%endif
%endif
BuildRequires: cuda-drivers
%endif

%if %{with amdgpu}
BuildRequires:  rock-dkms
%endif

BuildRequires:  %kernel_module_package_buildreqs

# Generate a preamble that gets attached to the kmod RPM(s). Kernel module
# dependencies can be declared here. The 'Obsoletes' and 'Provides' lines for
# RHEL allow the package to be referred to by its base name without having to
# explicitly specify a kernel version.
%(/bin/echo -e "\
Requires:       cray-hms-firmware \n\
Requires:       cray-libcxi-retry-handler \n\
%if 0%%{?rhel} \n\
Requires:       kmod-cray-slingshot-base-link \n\
Obsoletes:      kmod-%%{name} \n\
Provides:       kmod-%%{name} = %%version-%%release \n\
%else \n\
Requires:       cray-slingshot-base-link-kmp-%1 \n\
Requires:       sl-driver-kmp-%1 \n\
%endif" > %{_sourcedir}/%{name}.rpm_preamble)

%if 0%{with shasta_premium}
# The nvidia-gpu-build-obs package (necessary for building against CUDA
# drivers) causes a bogus default kernel flavor to be added. This causes
# builds to fail, as upstream dependencies (i.e. SBL) are not built for
# default on shasta-premium. Work around this by explicitly excluding the
# default flavor on shasta-premium
%kernel_module_package -x 64kb default %kmp_args
%else
%kernel_module_package %kmp_args
%endif

%description
Cassini driver

%package udev
Summary:    Udev rules for Cassini driver
Requires:   (%{distro_kernel_package_name} or %{name}-dkms)

%description udev
Udev rules for Cassini driver

%package devel
Summary:    Development files for Cassini driver
Requires:   cray-cassini-headers-user
Requires:   cray-slingshot-base-link-devel
Requires:   sl-driver-devel

%description devel
Development files for Cassini driver

%package dkms
Summary:    DKMS package for Cassini driver
BuildArch:  noarch
Requires:   dkms
Requires:   cray-cassini-headers-user
Requires:   cassini2-firmware-devel
Requires:   sl-driver-dkms
Requires:   sl-driver-devel
Requires:   cray-slingshot-base-link-dkms
Requires:   cray-slingshot-base-link-devel
Conflicts:  kmod-%name
Conflicts:  %name-kmp

%description dkms
DKMS support for Cassini driver

%prep
%setup
set -- *
mkdir source
mv "$@" source/
mkdir obj

%build
for flavor in %flavors_to_build; do
    rm -rf obj/$flavor
    cp -r source obj/$flavor

    make -C %{kernel_source $flavor} modules \
        M=$PWD/obj/$flavor/drivers/net/ethernet/hpe/ss1 \
        NO_BUILD_TESTS=1 \
        NO_SRIOV=1 \
        CASSINI_HEADERS_DIR=%{_includedir} \
        FIRMWARE_CASSINI_DIR=%{_includedir} \
        SLINGSHOT_BASE_LINK_DIR=%{_includedir} \
        SL_DIR=%{_includedir} \
        KBUILD_EXTRA_SYMBOLS="%{prefix}/src/slingshot-base-link/$flavor/Module.symvers \
            %{prefix}/src/sl/$flavor/Module.symvers" \
        %{?_smp_mflags}
done


%install
export INSTALL_MOD_PATH=$RPM_BUILD_ROOT
export INSTALL_MOD_DIR=extra/%{name}
for flavor in %flavors_to_build; do
    make -C %{kernel_source $flavor} M=$PWD/obj/$flavor/drivers/net/ethernet/hpe/ss1 NO_BUILD_TESTS=1 modules_install
    install -D $PWD/obj/$flavor/drivers/net/ethernet/hpe/ss1/Module.symvers $RPM_BUILD_ROOT/%{prefix}/src/cxi/$flavor/Module.symvers
done

# Remove any test modules (test-atu.ko, test-domain.ko, etc.) that got installed
rm -rf $INSTALL_MOD_PATH/lib/modules/*/$INSTALL_MOD_DIR/tests/

install -D source/include/linux/hpe/cxi/cxi.h %{buildroot}/%{_includedir}/linux/hpe/cxi/cxi.h
install -D source/include/uapi/ethernet/cxi-abi.h %{buildroot}/%{_includedir}/uapi/ethernet/cxi-abi.h
install -D source/include/uapi/misc/cxi.h %{buildroot}/%{_includedir}/uapi/misc/cxi.h
install -D --mode=0644 --target-directory=%{buildroot}/%{_udevrulesdir} source/50-cxi-driver.rules

%if 0%{?rhel}
# Centos/Rocky/RHEL does not exclude the depmod-generated modules.* files from
# the RPM, causing file conflicts when updating
find $RPM_BUILD_ROOT -iname 'modules.*' -exec rm {} \;
%endif

# DKMS bits
mkdir -p %{buildroot}/%{dkms_source_dir}
cp -r source/* %{buildroot}/%{dkms_source_dir}

sed \
    -e 's/@PACKAGE_NAME@/%{name}/g' \
    -e 's/@PACKAGE_VERSION@/%{version}-%{release}/g' \
    -e 's,@SHS_DKMS_AUX_DIR@,/etc/slingshot/dkms.conf.d,g' \
    %{buildroot}/%{dkms_source_dir}/dkms.conf.in \
    > %{buildroot}/%{dkms_source_dir}/dkms.conf

rm %{buildroot}/%{dkms_source_dir}/dkms.conf.in

sed \
    -e 's/@PACKAGE_NAME@/%{name}/g' \
    -e 's/@PACKAGE_VERSION@/%{version}-%{release}/g' \
    -e 's,@SHS_DKMS_AUX_DIR@,/etc/slingshot/dkms.conf.d,g' \
    %{buildroot}/%{dkms_source_dir}/dkms-aux-template.in \
    > %{buildroot}/%{dkms_source_dir}/dkms-aux-template

rm %{buildroot}/%{dkms_source_dir}/dkms-aux-template.in

echo "%dir %{dkms_source_dir}" > dkms-files
echo "%{dkms_source_dir}" >> dkms-files

%pre dkms

%post dkms
if [ -f /usr/libexec/dkms/common.postinst ] && [ -x /usr/libexec/dkms/common.postinst ]
then
    postinst=/usr/libexec/dkms/common.postinst
elif [ -f /usr/lib/dkms/common.postinst ] && [ -x /usr/lib/dkms/common.postinst ]
then
    postinst=/usr/lib/dkms/common.postinst
else
    echo "ERROR: did not find DKMS common.postinst" >&2
    exit 1
fi

${postinst} %{name} %{version}-%{release}

%preun dkms
# 'dkms remove' may fail in some cases (e.g. if the user has already run 'dkms
# remove'). Allow uninstallation to proceed even if it fails.
/usr/sbin/dkms remove -m %{name} -v %{version}-%{release} --all --rpm_safe_upgrade || true

%triggerin dkms -- cuda-drivers, intel-dmabuf-peer-mem-dkms, amdgpu-dkms
# Trigger dkms common.postinst to rebuild the DKMS module if a new GPU driver is
# installed.

# Skip if DKMS package is being uninstalled in this transaction
if [ $1 -eq 0 ]; then exit 0; fi

# Skip if triggering package is being upgraded, since %triggerpostun for the old
# package will run after %triggerin for the new package.
if [ $2 -ge 2 ]; then exit 0; fi

if [ -f /usr/libexec/dkms/common.postinst ] && [ -x /usr/libexec/dkms/common.postinst ]
then
    postinst=/usr/libexec/dkms/common.postinst
elif [ -f /usr/lib/dkms/common.postinst ] && [ -x /usr/lib/dkms/common.postinst ]
then
    postinst=/usr/lib/dkms/common.postinst
else
    echo "ERROR: did not find DKMS common.postinst" >&2
    exit 1
fi

${postinst} %{name} %{version}-%{release}

%triggerpostun dkms -- cuda-drivers, intel-dmabuf-peer-mem-dkms, amdgpu-dkms
# Trigger dkms common.postinst to rebuild the DKMS module after a GPU driver is
# uninstalled.

# Skip trigger if DKMS package is being uninstalled in this transaction
if [ $1 -eq 0 ]; then exit 0; fi

if [ -f /usr/libexec/dkms/common.postinst ] && [ -x /usr/libexec/dkms/common.postinst ]
then
    postinst=/usr/libexec/dkms/common.postinst
elif [ -f /usr/lib/dkms/common.postinst ] && [ -x /usr/lib/dkms/common.postinst ]
then
    postinst=/usr/lib/dkms/common.postinst
else
    echo "ERROR: did not find DKMS common.postinst" >&2
    exit 1
fi

${postinst} %{name} %{version}-%{release}

%files udev
%{_udevrulesdir}/50-cxi-driver.rules

%files devel
%{_includedir}/linux/hpe/cxi/cxi.h
%{_includedir}/uapi/ethernet/cxi-abi.h
%{_includedir}/uapi/misc/cxi.h
%{prefix}/src/cxi/*/Module.symvers

%files dkms -f dkms-files

%changelog
