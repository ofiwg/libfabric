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

use File::Basename;
use Getopt::Long;

my $source_dir_arg;
my $download_dir_arg;
my $coverity_token_arg;
my $logfile_dir_arg;
my $no_git_actions_arg = 0;
my $help_arg;
my $verbose_arg;
my $debug_arg;

my $ok = Getopt::Long::GetOptions("source-dir=s" => \$source_dir_arg,
                                  "download-dir=s" => \$download_dir_arg,
                                  "coverity-token=s" => \$coverity_token_arg,
                                  "logfile-dir=s" => \$logfile_dir_arg,
                                  "no-git-actions!" => \$no_git_actions_arg,
                                  "verbose" => \$verbose_arg,
                                  "debug" => \$debug_arg,
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

$verbose_arg = 1
    if ($debug_arg);

#####################################################################

sub doit {
    my $allowed_to_fail = shift;
    my $cmd = shift;
    my $stdout_file = shift;

    # Redirect stdout if requested
    if (defined $stdout_file) {
        $stdout_file = "$logfile_dir_arg/$stdout_file.log";
        unlink($stdout_file);
        $cmd .= " >$stdout_file";
    } elsif (!$verbose_arg && $cmd !~ />/) {
        $cmd .= " >/dev/null";
    }
    $cmd .= " 2>&1";

    my $rc = system($cmd);
    if (0 != $rc && !$allowed_to_fail) {
        # If we die/fail, ensure to a) restore the git tree to a clean
        # state, and b) change out of the temp tree so that it can be
        # removed upon exit.
        chdir($source_dir_arg);
        system("git clean -dfx");
        system("git reset --hard HEAD");
        chdir("/");

        die "Command $cmd failed: exit status $rc";
    }
    system("cat $stdout_file")
        if ($debug_arg && defined($stdout_file) && -f $stdout_file);
}

sub verbose {
    print @_
        if ($verbose_arg);
}

#####################################################################

chdir($source_dir_arg);

if (!$no_git_actions_arg) {
    # Git pull to get the latest; ensure we have a totally clean tree
    verbose("*** Ensuring we have a clean git tree...\n");
    doit(0, "git clean -dfx", "git-clean");
    doit(0, "git reset --hard HEAD", "git-reset");
    doit(0, "git pull", "git-pull");
}

# Get a git describe id (minus the initial 'v' in the tag name, if any)
my $gd = `git describe --tags --always`;
chomp($gd);
$gd =~ s/^v//;
verbose("*** Git describe: $gd\n");

# Read in configure.ac
verbose("*** Reading version number from configure.ac...\n");
open(IN, "configure.ac") || die "Can't open configure.ac for reading";
my $config;
$config .= $_
    while(<IN>);
close(IN);

# Get the original version number
$config =~ m/AC_INIT\(\[libfabric\], \[(.+?)\]/;
my $orig_version = $1;
verbose("*** Replacing configure.ac version: $orig_version\n");
my $version = $gd;
$version =~ y/-/./;
verbose("*** Nightly tarball version: $version\n");

# Is there already a tarball of this version in the download
# directory?  If so, just exit now without doing anything.
if (-f "$download_dir_arg/libfabric-$version.tar.gz") {
    verbose("*** Target tarball already exists: libfabric-$version.tar.gz\n");
    verbose("*** Exiting without doing anything\n");
    exit(0);
}

# Update the version number with the output from "git describe"
verbose("*** Re-writing configure.ac with git describe results...\n");
$config =~ s/(AC_INIT\(\[libfabric\], \[).+?\]/$1$version]/;
open(OUT, ">configure.ac");
print OUT $config;
close(OUT);

# Now make the tarball
verbose("*** Running autogen.sh...\n");
doit(0, "./autogen.sh", "autogen");
verbose("*** Running configure...\n");
doit(0, "./configure", "configure");

# Note that distscript.pl, invoked by "make dist", checks for a dirty
# git tree.  We have to tell it that a modified configure.ac is ok.
# So take the sha1sum of configure.ac and put it in a magic
# environment variable.
my $sha1 = `sha1sum configure.ac`;
chomp($sha1);
$ENV{'LIBFABRIC_DISTSCRIPT_SHA1_configure.ac'} = $sha1;

verbose("*** Running make distcheck...\n");
doit(0, "AM_MAKEFLAGS=-j32 make distcheck", "distcheck");

delete $ENV{'LIBFABRIC_DISTSCRIPT_SHA1_configure.ac'};

# Restore configure.ac
verbose("*** Restoring configure.ac...\n");
doit(0, "git checkout configure.ac");

# Move the resulting tarballs to the downloads directory
verbose("*** Placing tarballs in download directory...\n");
doit(0, "mv libfabric-$version.tar.gz libfabric-$version.tar.bz2 $download_dir_arg");

# Make sym links to these newest tarballs
chdir($download_dir_arg);
unlink("libfabric-latest.tar.gz");
unlink("libfabric-latest.tar.bz2");
doit(0, "ln -s libfabric-$version.tar.gz libfabric-latest.tar.gz");
doit(0, "ln -s libfabric-$version.tar.bz2 libfabric-latest.tar.bz2");

# Re-generate hashes
verbose("*** Re-generating md5/sha1sums...\n");
doit(0, "md5sum libfabric*tar* > md5sums.txt");
doit(0, "sha1sum libfabric*tar* > sha1sums.txt");

# Re-write latest.txt
verbose("*** Re-creating latest.txt...\n");
unlink("latest.txt");
open(OUT, ">latest.txt") || die "Can't write to latest.txt";
print OUT "libfabric-$version\n";
close(OUT);

# Run the coverity script if requested
if (defined($coverity_token_arg)) {
    verbose("*** Perparing/submitting to Coverity...\n");

    # The coverity script will be in the same directory as this script
    my $dir = dirname($0);
    my $cmd = "$dir/cron-submit-coverity.pl " .
        "--filename $download_dir_arg/libfabric-$version.tar.bz2 " .
        "--coverity-token $coverity_token_arg " .
        "--make-args=-j8 " .
        "--configure-args=\"--enable-sockets --enable-verbs --enable-psm --enable-usnic\" ";

    $cmd .= "--verbose "
        if ($verbose_arg);
    $cmd .= "--debug "
        if ($debug_arg);
    $cmd .= "--logfile-dir=$logfile_dir_arg"
        if (defined($logfile_dir_arg));

    # Coverity script will do its own logging
    doit(0, $cmd);
}

# All done
verbose("*** All done / nightly tarball\n");
exit(0);
