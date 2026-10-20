#!/bin/bash
# Run one local, verification-only encode for an A/B arm.
#
# Usage: run_one.sh <arm> <encoder_worktree> <citation> <out_dir>
# Env:   LOCAL_AB_PINS  directory with corpus/, engine/, rulespec-us/ and the
#                       three public keys read by local_encode.py (required)
#
# The encode runs under `env -i` with only HOME, USER, LOGNAME, a per-run
# TMPDIR, LANG and PATH, so no provider, R2 or Supabase key can reach it.
# Writes <out_dir>/run.json with arm, citation, exit code, wall seconds and
# the encoder commit.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
pins="${LOCAL_AB_PINS:?set LOCAL_AB_PINS to the pinned corpus/engine/rulespec-us directory}"
arm=$1; wt=$2; cit=$3; out=$4
rm -rf "$out"; mkdir -p "$out/tmp"
start=$(date +%s)
env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" TMPDIR="$out/tmp/" LANG=en_US.UTF-8 \
  PATH="$wt/.venv/bin:$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  LOCAL_AB_ATTEMPT_LOG="$out/failed_attempts.jsonl" \
  "$wt/.venv/bin/python" -I -u "$here/local_encode.py" "$pins" -- encode "$cit" \
    --backend codex \
    --corpus-path "$pins/corpus" \
    --axiom-rules-engine-path "$pins/engine" \
    --policy-repo-path "$pins/rulespec-us" \
    --output "$out/generated" \
    --emit-final-rejected-candidate "$out/final-rejected-candidate" \
    --mode repo-augmented \
    --require-complete-source-unit \
    --db "$out/encodings.db" \
    --no-sync \
  > "$out/stdout.log" 2> "$out/stderr.log"
rc=$?
end=$(date +%s)
printf '{"side":"%s","citation":"%s","rc":%d,"wall_s":%d,"encoder_commit":"%s"}\n' \
  "$arm" "$cit" $rc $((end-start)) "$(git -C "$wt" rev-parse HEAD)" > "$out/run.json"
