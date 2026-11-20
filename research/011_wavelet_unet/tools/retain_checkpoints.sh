#!/usr/bin/env bash
set -euo pipefail

run=$(realpath "$1")
snapshot=$(realpath "$2")
shift 2
deadline=$((SECONDS + 21600))

for arm in control candidate; do
    for step in "$@"; do
        destination="$run/retained_$step"
        mkdir -p "$destination"
        [[ ! -e "$destination/$arm.pth" ]] || continue
        name=$(sed -n 's/^name: //p' "$run/${arm}_train.yml")
        checkpoint="$snapshot/experiments/$name/models/net_g_$step.pth"
        state="$snapshot/experiments/$name/training_states/$step.state"
        until [[ -s "$checkpoint" && -s "$state" ]]; do
            if (( SECONDS >= deadline )) || [[ -s "$run/run/$arm.exit" ]]; then
                printf '%s: %s checkpoint %s unavailable\n' "$(date -Is)" "$arm" "$step" >&2
                exit 1
            fi
            sleep 15
        done
        cp "$checkpoint" "$destination/$arm.pth.tmp"
        cmp "$checkpoint" "$destination/$arm.pth.tmp"
        mv -n "$destination/$arm.pth.tmp" "$destination/$arm.pth"
        printf '%s: retained %s @ %s\n' "$(date -Is)" "$arm" "$step"
    done
done
for step in "$@"; do
    destination="$run/retained_$step"
    sha256sum "$destination"/{control,candidate}.pth >"$destination/sha256.txt"
done
