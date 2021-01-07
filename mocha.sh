#!/usr/bin/env bash
CURDIR=`dirname $0`

export NODE_ENV=test
$CURDIR/node_modules/mocha/bin/mocha --exit $@
