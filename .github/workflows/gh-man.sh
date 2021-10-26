#!/bin/bash

set -euxo pipefail

# Only do anything meaningful in the main ofiwg/libfabric repo.  If
# we're not in that repo, then don't do anything because it confuses
# people who fork the libfabric repo if GH Man Page Updater commits
# show up in their fork with different git hashes than the GH Man
# Page Updater commits on the ofiwg/libfabric repo.
if test -n "$REPO"; then
    first=`echo $REPO | cut -d/ -f1`
    second=`echo $REPO | cut -d/ -f2`

    if test "$first" != "ofiwg" -o "$second" != "libfabric"; then
        cat <<EOF

The GH Man Page Updater is contractually obligated to only operate on
the ofiwg/libfabric repository.

Exiting without doing anything.

EOF
        exit 0
    fi
fi

if test "$BASE_REF" != "main"; then
        echo "Not the main branch -- nothing to do!"
    exit 0
fi

git config --global user.name "OFIWG Bot"
git config --global user.email "ofiwg@lists.openfabrics.org"

mkdir tmp && cp man/*.md tmp/

branch_name=pr/update-gh-man-pages
git fetch origin gh-pages
git checkout gh-pages
git checkout -b $branch_name
cp tmp/*.md main/man/
git add main/man/*.md

set +e
git commit -as -m 'Update GH man pages'
st=$?
set -e

if test $st -ne 0; then
    echo "Nothing to commit -- nothing to do!"
    exit 0
fi

# Yes, we committed something.  Push the branch and make a PR.
# Extract the PR number.
git push --set-upstream origin $branch_name
url=`hub pull-request -b gh-pages -m 'Update GH man pages'`
pr_num=`echo $url | cut -d/ -f7`

# Wait for the required "DCO" CI to complete
i=0
sleep_time=5
max_seconds=300
i_max=`expr $max_seconds / $sleep_time`

echo "Waiting up to $max_seconds seconds for DCO CI to complete..."
while test $i -lt $i_max; do
    date
    set +e
    status=`hub ci-status --format "%t %S%n" | egrep '^DCO' | awk '{ print $2 }'`
    set -e
    if test "$status" = "success"; then
        echo "DCO CI is complete!"
        break
    fi
    sleep $sleep_time
    i=`expr $i + 1`
done

status=0
if test $i -lt $i_max; then
    # Sadly, there is no "hub" command to merge a PR.  So do it by
    # hand.
    curl \
        -XPUT \
        -H "Authorization: token $GITHUB_TOKEN" \
        https://api.github.com/repos/$GITHUB_REPOSITORY/pulls/$pr_num/merge
else
    echo "Sad panda; DCO CI didn't complete -- did not merge $url"
    status=1
fi

# Delete the remote branch
git push origin --delete $branch_name
exit $status
