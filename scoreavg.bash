#!/bin/bash

gawk -F'\t' '{sum+=$5; ++n} END { print "Avg. accuracy: "sum"/"n" = "sum/n }'
