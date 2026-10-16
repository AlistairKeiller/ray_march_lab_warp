#!/bin/bash
# Run every reference solution and check that it reports "complete ✅".
#   bash solutions/run.sh        # all puzzles
#   bash solutions/run.sh p03    # one puzzle
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

pass=0
fail=0
failed=()
for f in solutions/p*/p*.py; do
    p="$(basename "${f%.py}")"
    [ $# -gt 0 ] && [ "$p" != "$1" ] && continue
    printf '%s ... ' "$p"
    if out="$(uv run python "$f" 2>&1)" && grep -q "complete ✅" <<<"$out"; then
        echo "✓"
        pass=$((pass + 1))
    else
        echo "✗"
        tail -20 <<<"$out" | sed 's/^/    /'
        fail=$((fail + 1))
        failed+=("$p")
    fi
done

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] || { printf '  %s\n' "${failed[@]}"; exit 1; }
