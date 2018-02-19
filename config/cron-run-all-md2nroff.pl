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
use JSON;
use Data::Dumper;

my $repo_arg;
my $source_branch_arg;
my $pages_branch_arg;
my $logfile_dir_arg = "/tmp";
my $pat_file_arg;
my $verbose_arg;
my $help_arg;

my $ok = Getopt::Long::GetOptions("repo=s" => \$repo_arg,
                                  "source-branch=s" => \$source_branch_arg,
                                  "pages-branch=s" => \$pages_branch_arg,
                                  "logfile-dir=s" => \$logfile_dir_arg,
                                  "pat=s" => \$pat_file_arg,
                                  "help|h" => \$help_arg,
                                  "verbose" => \$verbose_arg,
                                  );

# Sanity checks
die "Must specify a git repo"
    if (!defined($repo_arg));
die "Must specify a git source branch"
    if (!defined($source_branch_arg));
die "Must specify a Github Personal Access Token (PAT) file"
    if (!defined($pat_file_arg));
die "Github Personal Access Token (PAT) file unreadable"
    if (! -r $pat_file_arg);

#####################################################################

open(FILE, $pat_file_arg) || die "Can't open Github Personal Access Token (PAT) file";
my $pat = <FILE>;
chomp($pat);
close(FILE);

$repo_arg =~ m/:(.+)\/(.+)\.git$/;
my $gh_org = $1;
my $gh_repo = $2;

#####################################################################

my $logfile_dir = $logfile_dir_arg;
my $logfile_counter = 1;

sub doit {
    my $allowed_to_fail = shift;
    my $cmd = shift;
    my $stdout_file = shift;

    # Redirect stdout if requested
    if (defined $stdout_file) {
        # Put a prefix on the logfiles so that we know that they
        # belong to this script, and put a counter so that we know the
        # sequence of logfiles
        $stdout_file = "runall-md2nroff-$logfile_counter-$stdout_file";
        ++$logfile_counter;

        $stdout_file = "$logfile_dir/$stdout_file.log";
        unlink($stdout_file);
        $cmd .= " >$stdout_file";
    } elsif (!$verbose_arg && $cmd !~ />/) {
        $cmd .= " >/dev/null";
    }
    $cmd .= " 2>&1";

    my $rc = system($cmd);
    if (0 != $rc && !$allowed_to_fail) {
        my_die("Command $cmd failed: exit status $rc");
    }

    system("cat $stdout_file")
        if ($verbose_arg && defined($stdout_file) && -f $stdout_file);
}

sub verbose {
    print @_
        if ($verbose_arg);
}

sub my_die {
    # Move out of our current cwd so that temp directories can be
    # automatically cleaned up at close.
    chdir("/");

    die @_;
}

sub read_json_file {
    my $filename = shift;
    my $unlink_file = shift;

    open(FILE, $filename);
    my $contents;
    while (<FILE>) {
        $contents .= $_;
    }
    close(FILE);

    unlink($filename)
        if ($unlink_file);

    return decode_json($contents);
}

#####################################################################

# Setup a logfile dir just for this run
my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) =
    localtime(time);
$logfile_dir =
    sprintf("%s/cron-run-all-md2nroff-logs-%04d-%02d-%02d-%02d%02d",
            $logfile_dir_arg, $year + 1900, $mon + 1, $mday,
            $hour, $min);
my $rc = system("mkdir $logfile_dir");
if ($rc != 0 || ! -d $logfile_dir || ! -w $logfile_dir) {
    my_die "mkdir of $logfile_dir failed, or can't write to it";
}

my $tmpdir = File::Temp->newdir();
verbose("*** Working in: $tmpdir\n");
chdir($tmpdir);

# First, git clone the source branch of the repo
verbose("*** Cloning repo: $repo_arg / $source_branch_arg...\n");
doit(0, "git clone --single-branch --branch $source_branch_arg $repo_arg source", "git-clone-source");

# Next, git clone the pages branch of repo
if (defined($pages_branch_arg)) {
    verbose("*** Cloning repo: $repo_arg / $pages_branch_arg...\n");
    doit(0, "git clone --single-branch --branch $pages_branch_arg $repo_arg pages", "git-clone-pages");
}

#####################################################################

# Find all the *.\d.md files in the source repo
verbose("*** Finding markdown man pages...\n");
opendir(DIR, "source/man");
my @markdown_files = grep { /\.\d\.md$/ && -f "source/man/$_" } readdir(DIR);
closedir(DIR);
verbose("Found: @markdown_files\n");

#####################################################################

# Copy each of the markdown files to the pages branch checkout
if (defined($pages_branch_arg)) {
    chdir("pages/master");
    foreach my $file (@markdown_files) {
        doit(0, "cp ../../source/man/$file man/$file", "loop-cp");

        # Is there a new man page?  If so, we need to "git add" it.
        my $out = `git status --porcelain man/$file`;
        doit(0, "git add man/$file", "loop-git-add")
            if ($out =~ /^\?\?/);
    }

    # Generate a new index.md with all the files that we just
    # published.  First, read in the header stub.
    open(IN, "man/index-head.txt") ||
        my_die("failed to open index-head.txt");
    my $str;
    $str .= $_
        while (<IN>);
    close(IN);

    # Write out the header stub into index.md itself
    open(OUT, ">man/index.md") ||
        my_die("failed to write to new index.md file");
    print OUT $str;

    # Now write out all the pages
    my @headings;
    push(@headings, { section=>7, title=>"General information" });
    push(@headings, { section=>3, title=>"API documentation" });
    foreach my $h (@headings) {
        print OUT "\n* $h->{title}\n";
        foreach my $file (sort(@markdown_files)) {
            if ($file =~ /\.$h->{section}\.md$/) {
                $file =~ m/^(.+)\.$h->{section}\.md$/;
                my $base = $1;
                print OUT "  * [$base($h->{section})]($base.$h->{section}.html)\n";
            }
        }
    }
    close(OUT);

    # Git commit those files in the pages repo and push them to the
    # upstream repo so that they go live.  If nothing changed, the commit
    # and push will be no-ops.
    chdir("..");
    doit(1, "git commit -s --no-verify -a -m \"Updated Markdown man pages from $source_branch_arg\"",
         "git-commit-pages");
    doit(1, "git push", "git-push-pages");
}

#####################################################################

# Now process each of the Markdown files in the source repo and
# generate new nroff man pages.
chdir("$tmpdir/source/man");
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
# them back upstream.  If nothing changed, these will be no-ops.  Note
# that there are mandatory CI checks on master, which means we can't
# push directly.  Instead, we must make a pull request.  Hence, don't
# git commit directly to the pages branch here; make a branch and
# commit there.

# Try to delete the old pr branch first (it's ok to fail -- i.e., if
# it wasn't there).
my $pr_branch_name = "pr/update-nroff-generated-man-pages";
doit(1, "git branch -D $pr_branch_name");
doit(0, "git checkout -b $pr_branch_name");

# Do the commit.  Save the git HEAD hash before and after so that we
# can tell if the "git commit" command actually resulted in a new
# commit.
my $old_head=`git rev-parse HEAD`;
doit(1, "git commit -s --no-verify -a -m \"Updated nroff-generated man pages\"",
     "git-commit-source-generated-man-pages");
my $new_head=`git rev-parse HEAD`;

# See if the commit was a no op or not.
if ($old_head ne $new_head) {
    chomp($new_head);

    # Push the branch up to github
    doit(0, "git push --force", "git-push-source-generated-man-pages");

    # Get the list of files
    open(GIT, 'git diff-tree --no-commit-id --name-only -r HEAD|') ||
        my_die "Cannot git diff-tree";
    my @files;
    while (<GIT>) {
        chomp;
        push(@files, $_);
    }
    close(GIT);

    # Create a new pull request
    my $cmd_base = "curl ";
    $cmd_base .= "-H 'Content-Type: application/json' ";
    $cmd_base .= "-H 'Authorization: token $pat' ";
    $cmd_base .= "-H 'User-Agent: OFIWG-bot' ";

    my $outfile = 'curl-out.json';
    unlink($outfile);

    my $body;
    $body = "The Nroff Elves created these man pages, just for you:\n\n";
    foreach my $f (@files) {
        $body .= "* `$f`\n";
    }

    my $json = {
        title => 'Update nroff-generated man pages',
        body  => $body,
        head  => $pr_branch_name,
        base  => 'master',
    };
    my $json_encoded = encode_json($json);

    my $cmd = $cmd_base;
    $cmd .= "--request POST ";
    $cmd .= "--data '$json_encoded' ";
    $cmd .= "https://api.github.com/repos/$gh_org/$gh_repo/pulls ";
    $cmd .= "-o $outfile";
    doit(0, $cmd, "github-create-pr");

    # Read the resulting file to find whether the PR creation
    # succeeded, and if so, what the URL of the new PR is.
    $json = read_json_file($outfile, 1);
    if (!exists($json->{'id'}) || !exists($json->{'url'})) {
        my_die "Failed to create PR";
    }

    my $pr_url = $json->{'url'};
    my $pr_num = $json->{'number'};
    verbose("Created PR #$pr_num\n");

    # Wait for the required DCO CI to complete on the git hash for the
    # latest commit.
    $outfile = "github-ci-status-check.json";

    $cmd = $cmd_base;
    $cmd .= "-o $outfile ";
    $cmd .= "https://api.github.com/repos/$gh_org/$gh_repo/commits/$new_head/statuses";

    my $count = 0;
    my $max_count = 30;
    my $happy = 0;
    verbose("Waiting for DCO CI to complete\n");

    # Only wait for $max_count iterations
    while (!$happy && $count < $max_count) {
        # Give the DCO hook time to run
        sleep(1);

        unlink($outfile);
        doit(0, $cmd, "github-check-ci-status");
        my $json = read_json_file($outfile, 1);

        if ($json and $#{$json} >= 0) {
            # If we got any statuses back, check them to see if we can
            # find a successful DCO signoff.  That would indicate that
            # the required CI test run.
            foreach my $j (@$json) {
                if ($j->{"context"} eq "DCO") {
                    verbose("Found DCO status on SHA $new_head\n");
                    if ($j->{"state"} eq "success") {
                        verbose("DCO is happy!\n");
                        $happy = 1;
                        last;
                    }
                }
            }
        }

        $count += 1;
    }

    my_die("Could not find a happy DCO status on $new_head")
        if (!$happy);

    # If we get here, it means the DCO CI is done/happy, so we can
    # merge the PR.
    $json = {
        commit_title   => "Merge pull request #$pr_num",
        commit_message => "More tasty nroff man pages for you, fresh out of the oven!",
        sha            => $new_head,
        merge_method   => "merge",
    };
    $json_encoded = encode_json($json);

    $outfile = "github-per-merge.json";
    unlink($outfile);

    $cmd = $cmd_base;
    $cmd .= "--request PUT ";
    $cmd .= "--data '$json_encoded' ";
    $cmd .= "-o $outfile ";
    $cmd .= "$pr_url/merge";
    doit(0, $cmd, "github-create-pr");

    # Remove the remote branch
    doit(1, "git push origin --delete $pr_branch_name", 'git-remove-remote-branch');
}

# Delete the local pull request branch
doit(0, "git checkout master");
doit(1, "git branch -D $pr_branch_name");

# chdir out of the tmpdir so that it can be removed
chdir("/");

# If we get here, we finished successfully, so there's no need to keep
# the logfile dir around
system("rm -rf $logfile_dir");

exit(0);
