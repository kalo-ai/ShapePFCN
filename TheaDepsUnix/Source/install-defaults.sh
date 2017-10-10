#!/bin/bash

SCRIPT_DIR=`dirname "$0"`
BASE_DIR="`perl -e 'use Cwd "abs_path";print abs_path(shift)' "$SCRIPT_DIR"`"

# One per line, so it breaks on first error
# "${BASE_DIR}/install.sh" --with-boost $@
"${BASE_DIR}/install.sh" --with-lib3ds $@
"${BASE_DIR}/install.sh" --with-cluto $@
"${BASE_DIR}/install.sh" --with-freeimageplus $@
"${BASE_DIR}/install.sh" --with-arpack $@
"${BASE_DIR}/install.sh" --with-arpack++ $@
"${BASE_DIR}/install.sh" --with-opt++ $@
