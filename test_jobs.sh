#!/bin/bash
CURRENT_CPU_NUM=$(grep "cpu cores" /proc/cpuinfo | head -1 | awk '{print $4}')
MHA_NUM_JOBS=$(awk -v n=$CURRENT_CPU_NUM 'BEGIN {n=int(n*0.8); if(n>12) print 12; else print n}')
echo "CURRENT_CPU_NUM=$CURRENT_CPU_NUM"
echo "MHA_NUM_JOBS=$MHA_NUM_JOBS"