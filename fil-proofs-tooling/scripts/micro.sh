#!/usr/bin/env bash

MICRO_SDERR=$(mktemp)
MICRO_SDOUT=$(mktemp)
JQ_STDERR=$(mktemp)

CMD="cargo run --bin micro --release ${@}"

eval "RUST_BACKTRACE=1 RUSTFLAGS=\"-Awarnings -C target-cpu=native\" ${CMD}" 1>$MICRO_SDOUT 2>$MICRO_SDERR

MICRO_EXIT_CODE=$?

cat $MICRO_SDOUT | jq '.' 2>$JQ_STDERR

JQ_EXIT_CODE=$?

if [[ ! $MICRO_EXIT_CODE -eq 0 || ! $JQ_EXIT_CODE -eq 0 ]]; then
    echo >&2 "********************************************"
    echo >&2 "* micro failed - dumping debug information *"
    echo >&2 "********************************************"
    echo >&2 ""
    echo >&2 "<COMMAND>"
    echo >&2 "${CMD}"
    echo >&2 "</COMMAND>"
    echo >&2 ""
    echo >&2 "<MICRO_SDERR>"
    echo >&2 "$(cat $MICRO_SDERR)"
    echo >&2 "</MICRO_SDERR>"
    echo >&2 ""
    echo >&2 "<MICRO_SDOUT>"
    echo >&2 "$(cat $MICRO_SDOUT)"
    echo >&2 "</MICRO_SDOUT>"
    echo >&2 ""
    echo >&2 "<JQ_STDERR>"
    echo >&2 "$(cat $JQ_STDERR)"
    echo >&2 "</JQ_STDERR>"
    exit 1
fi
