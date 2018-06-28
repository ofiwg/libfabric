#!/bin/bash
# Install git hook to run checkpatch on commits

BASE_REPO_DIR="$(git rev-parse --show-toplevel)"

if [[ -e "${BASE_REPO_DIR}/.git/hooks/pre-commit" ]]; then
	echo "Error! The pre-commit git hook is already present."
	exit 1
fi

cat > "${BASE_REPO_DIR}/.git/hooks/pre-commit" << EOL
#!/usr/bin/env bash
exec git diff --cached | ./contrib/checkpatch.pl --no-tree --no-signoff
EOL
chmod +x "${BASE_REPO_DIR}/.git/hooks/pre-commit"

echo "The pre-commit git hook was installed successfully."
