#!/bin/sh
# SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
# Copyright 2020 Hewlett Packard Enterprise Development LP


#
# Install git hook to run checkpatch on commits
#

scriptname="${0##*/}"

set -e

BASE_REPO_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ $? -ne 0 ]
then
	printf '%s: Error! not in a git clone or repo.\n'  "${scriptname}"
	exit 1
fi
if [ ! -d "${BASE_REPO_DIR}/.git/hooks" ]
then
	printf '%s: Error! .git/hooks/ not found.\n'  "${scriptname}"
	exit 1
fi
if [ -e "${BASE_REPO_DIR}/.git/hooks/pre-commit" ]
then
	printf '%s: Error! The pre-commit git hook is already present.\n' \
		"${scriptname}"
	exit 1
fi

cat >"${BASE_REPO_DIR}/.git/hooks/pre-commit" <<'EOI'
#!/bin/sh

scriptname="${0##*/}"

#
# overall error value (0 --> no-error) returned by this script.
#
# The intention is to allow each 'hook' run gather its own success/failure
# setting errval to non-zero if this overall script should return error.
#
errval=0

if [ ${GIT_HOOKS_PRE_COMMIT_HPE_CRAY_BRANCH_CHECK_SKIP:-0} -eq 0 ]
then
	branchname=$(git symbolic-ref --short --quiet HEAD \
		|| ( git branch --list --format='%(refname:lstrip=2)' HEAD \
			| sed -e 's/(HEAD detached at \(.*\))/detached@\1/g' ))
	if [ "${branchname}" = master ]
	then
		errval=1
		printf '\n%s: ERROR:  commits to master are disallowed\n\n'\
			 "${scriptname}"
	fi
fi

if [ ${GIT_HOOKS_PRE_COMMIT_HPE_CRAY_CHECKPATCH_SKIP:-0} -eq 0 ]
then
	git diff --no-ext-diff --cached \
	| ./contrib/checkpatch.pl\
		--no-tree\
		--no-signoff\
		--ignore FILE_PATH_CHANGES,ENOSYS
	if [ $? -ne 0 ]
	then
		errval=1
	fi
fi

#
# indicate hook status
#
exit ${errval}
EOI
chmod +x "${BASE_REPO_DIR}/.git/hooks/pre-commit"

printf '%s: The pre-commit git hook was installed successfully.\n'\
	"${scriptname}"
exit 0
