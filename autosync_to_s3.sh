#!/bin/sh

for (( ; ; ))
do
   ./sync_data_to_s3.sh
   echo "Synced! Waiting..."
   sleep 600s
done
