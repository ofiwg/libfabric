%if 0%{?suse_version}
# Suse convention is to name static library RPM <packagename>-devel-static
%define static_subpackage devel-static
%else
# Redhat convension is <packagename>-static
%define static_subpackage static
%endif

Name:       cray-libcxi
Version:    1.0.2
Release:    %(echo ${BUILD_METADATA})
Summary:    Cassini userspace library
License:    Dual LGPL-2.1/BSD-3-Clause
Source0:    libcxi-%{version}.tar.gz
Prefix:     /usr

BuildRequires:  cray-cassini-headers-user
BuildRequires:  cray-cxi-driver-devel
BuildRequires:  cray-cassini-csr-defs
BuildRequires:  autoconf
BuildRequires:  automake
BuildRequires:  libtool
BuildRequires:  libconfig-devel
BuildRequires:  systemd-rpm-macros
BuildRequires:  libuv-devel
BuildRequires:  fuse-devel
BuildRequires:  systemd-devel
BuildRequires:  libyaml-devel
BuildRequires:  libnl3-devel
%if 0%{?rhel}
BuildRequires:  lm_sensors-devel
BuildRequires:  numactl-devel
Requires:  libnl3
%else
BuildRequires:  libsensors4-devel
BuildRequires:  libnuma-devel
Requires:  libnl3-200
%endif

%description
Cassini userspace library. 
Built using GPU's
%{expand:%(cat /var/tmp/gpu-versions)}

%package retry-handler
Summary:    Cassini retry handler
Requires:   cray-libcxi = %{version}-%{release}
Requires:   systemd-rpm-macros
%systemd_requires

%if 0%{?rhel}
Requires:   libconfig libuv fuse-libs
%else
Requires:   libconfig11 libuv1 libfuse2
%endif

%description retry-handler
Cassini retry handler

%package devel
Summary:    LibCXI development files
Requires:   cray-libcxi = %{version}-%{release}
Requires:   cray-cxi-driver-devel

%description devel
LibCXI development files

%package %static_subpackage
Summary:    LibCXI development files (static library)
Requires:   cray-libcxi-devel = %{version}-%{release}

%description %static_subpackage
LibCXI development files (static library)

%package utils
Summary:    LibCXI Utilities
Requires:   cray-libcxi = %{version}-%{release}
%if 0%{?rhel}
Requires:   lm_sensors-libs libyaml numactl-libs
%else
Requires:   libsensors4 libyaml-0-2 libnuma1
%endif

%description utils
LibCXI Utilities

%package dracut
Summary:        Dracut initramfs support for CXI software stack
Requires(pre):  cray-hms-firmware
Requires(pre):  cray-cxi-driver-udev
Requires(pre):  cray-libcxi-retry-handler
%if 0%{?sle_version}
Requires(pre):  (cray-cxi-driver-kmp or cray-cxi-driver-dkms)
Requires(pre):  (cray-slingshot-base-link-kmp or cray-slingshot-base-link-dkms)
Requires(pre):  (sl-driver-kmp or sl-driver-dkms)
%else
Requires(pre):  (kmod-cray-cxi-driver or cray-cxi-driver-dkms)
Requires(pre):  (kmod-cray-slingshot-base-link or cray-cxi-driver-dkms)
Requires(pre):  (kmod-sl-driver or sl-driver-dkms)
%endif

%description dracut
Dracut initramfs support for CXI software stack

%prep
%setup -n libcxi-%{version}

%build
./autogen.sh
%configure --with-systemd
make %{?_smp_mflags}

%install
%make_install

# Dynamic dracut config file to install RH bits into initrd.
echo "install_items+=\" %{_bindir}/cxi_rh \"" >> 50-cxi-rh.conf
echo "install_items+=\" %{_unitdir}/cxi_rh@.service \"" >> 50-cxi-rh.conf
echo "install_items+=\" %{_unitdir}/cxi_rh.target \"" >> 50-cxi-rh.conf
echo "install_items+=\" %{_sysconfdir}/cxi_rh.conf \"" >> 50-cxi-rh.conf
echo "install_items+=\" %{_udevrulesdir}/60-cxi.rules \"" >> 50-cxi-rh.conf

# Dynamic dracut config file to install libcxi.so into initrd.
for full_path_so in $(ls %{buildroot}/%{_libdir}/libcxi.so*); do
    so=$(basename ${full_path_so})
    echo "install_items+=\" %{_libdir}/${so} \"" >> 50-libcxi.conf
done

echo "enable cxi_rh.target" > 99-cxi_rh.preset

install -D --target-directory=%{buildroot}/etc/dracut.conf.d/ 50-cxi-rh.conf
install -D --target-directory=%{buildroot}/etc/dracut.conf.d/ 50-libcxi.conf
install -D --target-directory=%{buildroot}/etc/dracut.conf.d/ retry_handler/dracut.conf.d/50-cxi-driver.conf
install -D --target-directory=%{buildroot}/%{_presetdir}/ 99-cxi_rh.preset


%files
%{_libdir}/libcxi.so*

%exclude %{_bindir}/cxi_device_list
%exclude %{_bindir}/cxi_udp_gen
%exclude %{_bindir}/test_map_csr
%exclude %{_bindir}/test_write_csr
%exclude %{_mandir}/man1/cxi_udp_gen.1

%files retry-handler
%{_bindir}/cxi_rh
%{_unitdir}/cxi_rh@.service
%{_unitdir}/cxi_rh.target
%{_udevrulesdir}/60-cxi.rules
%{_presetdir}/99-cxi_rh.preset

%config(noreplace) %{_sysconfdir}/cxi_rh.conf

%files devel
%{_includedir}/libcxi/libcxi.h
%{_libdir}/pkgconfig/libcxi.pc

%files %static_subpackage
%{_libdir}/libcxi.a
%{_libdir}/libcxiutils.a

%files utils
%{_libdir}/libcxiutils.so*
%{_bindir}/cxi_send_lat
%{_bindir}/cxi_send_bw
%{_bindir}/cxi_heatsink_check
%{_bindir}/cxi_write_bw
%{_bindir}/cxi_write_lat
%{_bindir}/cxi_read_bw
%{_bindir}/cxi_read_lat
%{_bindir}/cxi_atomic_bw
%{_bindir}/cxi_atomic_lat
%{_bindir}/cxi_stat
%{_bindir}/cxi_service
%{_bindir}/cxi_gpu_loopback_bw
%{_bindir}/cxi_dump_csrs
%{_mandir}/man1/*
%{_mandir}/man7/*
%{_datadir}/cxi/cxi_service_template.yaml

%files dracut
%{_sysconfdir}/dracut.conf.d/*.conf

%exclude
%{_libdir}/libcxi.la
%{_libdir}/libcxiutils.la

%if 0%{?rhel}
%post retry-handler
%systemd_post cxi_rh@.service
%systemd_post cxi_rh.target

%preun retry-handler
%systemd_preun cxi_rh.target
%systemd_preun cxi_rh@.service

%postun retry-handler
%systemd_postun cxi_rh.target
%systemd_postun cxi_rh@.service

%else
%pre retry-handler
%service_add_pre cxi_rh@.service
%service_add_pre cxi_rh.target

%post retry-handler
%service_add_post cxi_rh@.service
%service_add_post cxi_rh.target

%preun retry-handler
%service_del_preun cxi_rh.target
%service_del_preun cxi_rh@.service

%postun retry-handler
%service_del_postun cxi_rh.target
%service_del_postun cxi_rh@.service
%endif

%postun dracut
# Remove firmware from initrd.
%if 0%{?rhel}
/usr/bin/dracut --force
%else
if test -x /usr/lib/module-init-tools/regenerate-initrd-posttrans; then
        mkdir -p /run/regenerate-initrd
        touch /run/regenerate-initrd/all
        /bin/bash -${-/e/} /usr/lib/module-init-tools/regenerate-initrd-posttrans
fi
%endif

%posttrans dracut
# Install firmware in initrd.
%if 0%{?rhel}
/usr/bin/dracut --force
%else
if test -x /usr/lib/module-init-tools/regenerate-initrd-posttrans; then
        mkdir -p /run/regenerate-initrd
        touch /run/regenerate-initrd/all
        /bin/bash -${-/e/} /usr/lib/module-init-tools/regenerate-initrd-posttrans
fi
%endif

%if 0%{?rhel}
%define dracut_triggers kmod-cray-slingshot-base-link, kmod-cray-cxi-driver, cray-cxi-driver-udev, cray-libcxi, cray-libcxi-retry-handler
%else
%define dracut_triggers cray-slingshot-base-link-kmp, cray-cxi-driver-kmp, cray-cxi-driver-udev, cray-libcxi, cray-libcxi-retry-handler
%endif

%triggerin -n cray-libcxi-dracut -- %dracut_triggers

%if 0%{?rhel}
/usr/bin/dracut --force
%else
if test -x /usr/lib/module-init-tools/regenerate-initrd-posttrans; then
        mkdir -p /run/regenerate-initrd
        touch /run/regenerate-initrd/all
        /bin/bash -${-/e/} /usr/lib/module-init-tools/regenerate-initrd-posttrans
fi
%endif

%triggerpostun -n cray-libcxi-dracut -- %dracut_triggers

%if 0%{?rhel}
/usr/bin/dracut --force
%else
if test -x /usr/lib/module-init-tools/regenerate-initrd-posttrans; then
        mkdir -p /run/regenerate-initrd
        touch /run/regenerate-initrd/all
        /bin/bash -${-/e/} /usr/lib/module-init-tools/regenerate-initrd-posttrans
fi
%endif

%changelog
