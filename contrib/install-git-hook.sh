#!/bin/bash

# Install git hook to run checkpatch on commits

if [[ ! -d .git ]]; then
	echo "Install script must be run from the project root: ./contrib/install-git-hook.sh"
	exit 1
fi

if [[ -e .git/hooks/post-commit ]]; then
	echo "post-commit git hooks already present"
	exit 1
fi

echo "exec git show --format=email HEAD | ./contrib/checkpatch.pl --no-tree --no-signoff" > .git/hooks/post-commit
chmod +x .git/hooks/post-commit

echo "Hook installed"
