#!/usr/bin/env perl

# Script to pull down the latest markdown man pages from the libfabric
# git repo.  Iterate over them, converting each to an nroff man page
# and also copying+committing them to the gh-pages branch.  Finally,
# git push them back upstream (so that Github will render + serve them
# up as web pages).

use strict;
use warnings;

use POSIX;
use File::Basename;
use Getopt::Long;
use File::Temp;

my $repo_arg;
my $source_branch_arg;
my $pages_branch_arg;
my $logfile_dir_arg;
my $verbose_arg;
my $help_arg;

my $ok = Getopt::Long::GetOptions("repo=s" => \$repo_arg,
                                  "source-branch=s" => \$source_branch_arg,
                                  "pages-branch=s" => \$pages_branch_arg,
                                  "logfile-dir=s" => \$logfile_dir_arg,
                                  "help|h" => \$help_arg,
                                  "verbose" => \$verbose_arg,
                                  );

# Sanity checks
die "Must specify a git repo"
    if (!defined($repo_arg));
die "Must specify a git source branch"
    if (!defined($source_branch_arg));
die "Must specify a git pages branch"
    if (!defined($pages_branch_arg));

#####################################################################

my $logfile_counter = 1;

sub doit {
    my $allowed_to_fail = shift;
    my $cmd = shift;
    my $stdout_file = shift;

    # Put a prefix on the logfiles so that we know that they belong to
    # this script, and put a counter so that we know the sequence of
    # logfiles
    $stdout_file = "runall-md2nroff-$logfile_counter-$stdout_file";
    ++$logfile_counter;

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
        # If we die/fail, ensure to change out of the temp tree so
        # that it can be removed upon exit.
        chdir("/");
        die "Command $cmd failed: exit status $rc";
    }

    system("cat $stdout_file")
        if ($verbose_arg && defined($stdout_file) && -f $stdout_file);
}

sub verbose {
    print @_
        if ($verbose_arg);
}

#####################################################################

# First, git clone the source branch of the repo
verbose("*** Cloning repo: $repo_arg / $source_branch_arg...\n");
my $tmpdir = File::Temp->newdir();

chdir($tmpdir);
doit(0, "git clone --single-branch --branch $source_branch_arg $repo_arg source", "git-clone");

# Next, git clone the pages branch of repo
verbose("*** Cloning repo: $repo_arg / $pages_branch_arg...\n");
doit(0, "git clone --single-branch --branch $pages_branch_arg $repo_arg pages", "git-clone2");

#####################################################################

# Find all the *.\d.md files in the source repo
verbose("*** Finding markdown man pages...\n");
opendir(DIR, "source/man");
my @markdown_files = grep { /\.\d\.md$/ && -f "source/man/$_" } readdir(DIR);
closedir(DIR);
verbose("Found: @markdown_files\n");

#####################################################################

# Copy each of the markdown files to the pages branch checkout
chdir("pages/master");
foreach my $file (@markdown_files) {
    doit(0, "cp ../../source/man/$file man/$file", "loop-cp");

    # Is there a new man page?  If so, we need to "git add" it.
    my $out = `git status --porcelain man/$file`;
    doit(0, "git add man/$file", "loop-git-add")
        if ($out =~ /^\?\?/);
}

# Git commit those files in the pages repo and push them to the
# upstream repo so that they go live.  If nothing changed, the commit
# and push will be no-ops.
chdir("..");
doit(1, "git commit --no-verify -a -m \"Updated Markdown man pages from $source_branch_arg\"",
     "git-commit-first");
doit(1, "git push", "git-push-first");

#####################################################################

# Now process each of the Markdown files in the source repo and
# generate new nroff man pages.
chdir("../source/man");
foreach my $file (@markdown_files) {
    doit(0, "../config/md2nroff.pl --source $file", "loop2-md2nroff");

    # Did we generate a new man page?  If so, we need to "git add" it.
    my $man_file = basename($file);

    $man_file =~ m/\.(\d)\.md$/;
    my $section = $1;

    $man_file =~ s/\.md$//;

    my $full_filename = "man$section/$man_file";

    my $out = `git status --porcelain $full_filename`;
    doit(0, "git add $full_filename", "loop2-git-add")
        if ($out =~ /^\?\?/);
}

# Similar to above: commit the newly-generated nroff pages and push
# them back upstream.  If nothing changed, these will be no-ops.
doit(1, "git commit --no-verify -a -m \"Updated nroff-generated man pages\"", "git-commit-final");
doit(1, "git push", "git-push-final");

# chdir out of the tmpdir so that it can be removed
chdir("/");

exit(0);
