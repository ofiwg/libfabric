#!/usr/bin/env perl

#
# Script to automate the steps to make pristine nightly tarballs and
# place them in a specified output directory (presumeably where the
# web server can serve them up to the world).  Designed to be invoked
# via cron (i.e., no output unless --verbose is specified).
#
# The specific steps are:
#
# 1. Ensure we have a pristine, clean git tree
# 2. git pull the latest down from upstream
# 3. Use "git describe" to get a unique string to represent this tarball
# 4. If we already have a tarball for this git HEAD in the destination
#    directory, no need to do anything further / quit
# 5. Re-write configure.ac to include the git describe string in the
#    version argument to AC_INIT
# 6. autogen.sh/configure/make distcheck
# 7. Move the resulting tarballs to the destination directory
# 8. Re-create sym links for libfabric-latest.tar.(gz|bz2)
# 9. Re-generate md5 and sh1 hash files
#
# Note that this script intentionally does *not* prune old nightly
# tarballs as result of an OFIWG phone discussion, the conlcusion of
# which was "What the heck; we have lots of disk space.  Keep
# everything."
#

use strict;
use warnings;

use Getopt::Long;

my $source_dir_arg;
my $download_dir_arg;
my $help_arg;
my $verbose_arg;

my $ok = Getopt::Long::GetOptions("source-dir=s" => \$source_dir_arg,
                                  "download-dir=s" => \$download_dir_arg,
                                  "verbose" => \$verbose_arg,
                                  "help|h" => \$help_arg,
                                  );

if ($help_arg || !$ok) {
    print "$0 --source-dir libfabric_git_tree --download-dir download_tree\n";
    exit(0);
}

# Sanity checks
die "Must specify both --source-dir and --download-dir"
    if (!defined($source_dir_arg) || $source_dir_arg eq "" ||
        !defined($download_dir_arg) || $download_dir_arg eq "");
die "$source_dir_arg is not a valid directory"
    if (! -d $source_dir_arg);
die "$source_dir_arg is not libfabric git clone"
    if (! -d "$source_dir_arg/.git" || ! -f "$source_dir_arg/src/fi_tostr.c");
die "$download_dir_arg is not a valid directory"
    if (! -d $download_dir_arg);

#####################################################################

sub doit {
    my $allowed_to_fail = shift;
    my $cmd = shift;

    $cmd .= " 2>/dev/null >/dev/null"
        if (!$verbose_arg);

    my $rc = system($cmd);
    if (0 != $rc && !$allowed_to_fail) {
        # If we die/fail, ensure to change out of the temp tree so
        # that it can be removed upon exit.
        chdir("/");
        die "Command @_ failed: exit status $rc";
    }
}

sub verbose {
    print @_
        if ($verbose_arg);
}

#####################################################################

# Git pull to get the latest; ensure we have a totally clean tree
verbose("*** Ensuring we have a clean git tree...\n");
chdir($source_dir_arg);
doit(0, "git clean -df");
doit(0, "git checkout .");
doit(0, "git pull");

# Get a git describe id
my $gd = `git describe --tags --always`;
chomp($gd);
verbose("*** Git describe: $gd\n");

# Read in configure.ac
verbose("*** Re-writing configure.ac with git describe results...\n");
open(IN, "configure.ac") || die "Can't open configure.ac for reading";
my $config;
$config .= $_
    while(<IN>);
close(IN);

# Get the original version number
$config =~ m/AC_INIT\(\[libfabric\], \[(.+?)\]/;
my $version = $1;

# Is there already a tarball of this version in the download
# directory?  If so, just exit now without doing anything.
if (-f "$download_dir_arg/libfabric-$version-$gd.tar.gz") {
    verbose("*** Target tarball already exists: libfabric-$version-$gd.tar.gz\n");
    verbose("*** Exiting without doing anything\n");
    exit(0);
}

# Update the version number with the output from "git describe"
$config =~ s/(AC_INIT\(\[libfabric\], \[.+?)\]/$1-$gd]/;
open(OUT, ">configure.ac");
print OUT $config;
close(OUT);

# Now make the tarball
verbose("*** Running autogen.sh...\n");
doit(0, "./autogen.sh");
verbose("*** Running configure...\n");
doit(0, "./configure");
verbose("*** Running make distcheck...\n");
doit(0, "AM_MAKEFLAGS=-j32 make distcheck");

# Restore configure.ac
verbose("*** Restoring configure.ac...\n");
doit(0, "git checkout configure.ac");

# Move the resulting tarballs to the downloads directory
verbose("*** Placing tarballs in download directory...\n");
doit(0, "mv libfabric-$version-$gd.tar.gz libfabric-$version-$gd.tar.bz2 $download_dir_arg");

# Make sym links to these newest tarballs
chdir($download_dir_arg);
unlink("libfabric-latest.tar.gz");
unlink("libfabric-latest.tar.bz2");
doit(0, "ln -s libfabric-$version-$gd.tar.gz libfabric-latest.tar.gz");
doit(0, "ln -s libfabric-$version-$gd.tar.bz2 libfabric-latest.tar.bz2");

# Re-generate hashes
verbose("*** Re-generating md5/sha1sums...\n");
doit(0, "md5sum libfabric*tar* > md5sums.txt");
doit(0, "sha1sum libfabric*tar* > sha1sums.txt");

# All done
exit(0);
