#!/usr/bin/env bash

which jq >/dev/null || {
    printf '%s\n' "error: jq" >&2
    exit 1
}

BENCHY_STDOUT=$(mktemp)
GTIME_STDERR=$(mktemp)
JQ_STDERR=$(mktemp)

GTIME_BIN="env time"
GTIME_ARG="-f '{ \"max-resident-set-size-kb\": %M }' cargo run --quiet --bin benchy --release -- ${@}"

if [[ $(env time --version 2>&1) != *"GNU"* ]]; then
    if [[ $(/usr/bin/time --version 2>&1) != *"GNU"* ]]; then
        if [[ $(env gtime --version 2>&1) != *"GNU"* ]]; then
            printf '%s\n' "error: GNU time not installed" >&2
            exit 1
        else
            GTIME_BIN="gtime"
        fi
    else
        GTIME_BIN="/usr/bin/time"
    fi
fi

CMD="${GTIME_BIN} ${GTIME_ARG}"

eval "RUST_BACKTRACE=1 RUSTFLAGS=\"-Awarnings -C target-cpu=native\" ${CMD}" >$BENCHY_STDOUT 2>$GTIME_STDERR

GTIME_EXIT_CODE=$?

jq -s '.[0] * .[1]' $BENCHY_STDOUT $GTIME_STDERR 2>$JQ_STDERR

JQ_EXIT_CODE=$?

if [[ ! $GTIME_EXIT_CODE -eq 0 || ! $JQ_EXIT_CODE -eq 0 ]]; then
    echo >&2 "*********************************************"
    echo >&2 "* benchy failed - dumping debug information *"
    echo >&2 "*********************************************"
    echo >&2 ""
    echo >&2 "<COMMAND>"
    echo >&2 "${CMD}"
    echo >&2 "</COMMAND>"
    echo >&2 ""
    echo >&2 "<GTIME_STDERR>"
    echo >&2 "$(cat $GTIME_STDERR)"
    echo >&2 "</GTIME_STDERR>"
    echo >&2 ""
    echo >&2 "<BENCHY_STDOUT>"
    echo >&2 "$(cat $BENCHY_STDOUT)"
    echo >&2 "</BENCHY_STDOUT>"
    echo >&2 ""
    echo >&2 "<JQ_STDERR>"
    echo >&2 "$(cat $JQ_STDERR)"
    echo >&2 "</JQ_STDERR>"
    exit 1
fi
