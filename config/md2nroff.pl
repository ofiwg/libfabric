#!/usr/bin/env perl

use strict;
use warnings;

use POSIX;
use File::Basename;
use Getopt::Long;
use File::Temp;

my $source_arg;
my $target_arg;
my $help_arg;

my $ok = Getopt::Long::GetOptions("source=s" => \$source_arg,
                                  "target=s" => \$target_arg,
                                  "help|h" => \$help_arg,
                                  );

if ($help_arg) {
    print "$0 --source input_MD_file --target output_nroff_file\n";
    exit(0);
}

# Sanity checks
die "Must specify both a source file"
    if (!defined($source_arg));
die "Source file does not exist ($source_arg)"
    if (! -r $source_arg);

my $pandoc = `which pandoc`;
die "Cannot find pandoc executable"
    if ($pandoc eq "");

#####################################################################

my $file = $source_arg;
$file =~ m/(\d+).md/;
my $section = $1;
die "Could not figure out the man page section: $source_arg"
    if (!defined($section));
my $shortfile = basename($file);
$shortfile =~ s/\.$section\.md$//;

# If the target file was not specified, derive it from the source file
if (!defined($target_arg)) {
    $target_arg = $source_arg;
    $target_arg =~ s/\.md$//;
}

print "*** Processing: $file -> $target_arg\n";

# Read in the file
my $str;
open(IN, $file)
    || die "Can't open $file";
$str .= $_
    while (<IN>);
close(IN);

# Remove the Jekyll header
$str =~ s/.*---\n.+?---\n//s;

# Remove the {% include ... %} directives
$str =~ s/\n{0,1}\s*{%\s+include .+?\s+%}\s*\n/\n/g;

# Change {% highlight c %} to ```c
$str =~ s/^\s*{%\s+highlight\s+c\s+%}\s*$/\n```c/gmi;

# Change {% endhighlight %} to ```
$str =~ s/^\s*\{\%\s+endhighlight\s+\%\}\s*$/```\n/gmi;

# Pandoc does not handle markdown links in output nroff properly,
# so just remove all links.
while ($str =~ m/\[(.+?)\]\(.+?\)/) {
    my $text = $1;
    $str =~ s/\[(.+?)\]\(.+?\)/$text/;
}

# Add the pando cheader
$str = "% $shortfile($section) Libfabric Programmer's Manual | \@VERSION\@
% OpenFabrics
% \@DATE\@\n\n$str";

# Now that we have the string result, is it different than the target
# file?
my $target_str;
my $file_open = 1;
$file_open = 0
    if (!open(IN, $target_arg));
if ($file_open) {
    $target_str .= $_
        while(<IN>);
    close(IN);
    $target_str =~ s/% OpenFabrics\n% \d\d\d\d-\d\d-\d\d\n\n/% OpenFabrics\n% \@DATE\@\n\n/;

    # If they're the same, then we're done with this file
    if ($target_str eq $str) {
        print "--> Files the same; not written\n";
        exit(0);
    }
}

# What's the date right now?
my $now_string = strftime "%Y-%m-%d", localtime;

# If we get here, we need to write a new target file
$str =~ s/\@DATE\@/$now_string/g;
open(OUT, "|pandoc -s --from=markdown --to=man -o $target_arg")
    || die "Can't run pandoc";
print OUT $str;
close(OUT);

print "--> Wrote new $target_arg\n";
exit(0);
