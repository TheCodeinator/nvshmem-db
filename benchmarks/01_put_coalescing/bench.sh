#!/bin/bash

# Only table size parameter for now

# Make sure one parameter is given, which is a positive integer

if [ $# -ne 1 ]; then
    echo "Usage: $0 <table_size>"
    exit 1
fi

if ! [[ $1 =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <table_size>"
    exit 1
fi

