#!/bin/bash
cd /DATA/swagata/akshat_ICLR
echo "-- disk --"
df -h /DATA | tail -n 1
echo "-- sd3 npz --"
find cfs_sd3_calib/latents -name '*.npz' 2>/dev/null | wc -l
echo "-- sd3 rows minus header --"
tail -n +2 cfs_sd3_calib/concept_metrics.csv | wc -l
echo "-- distinct sd3 pairs in csv --"
tail -n +2 cfs_sd3_calib/concept_metrics.csv | cut -d, -f1,4 | sort -u | wc -l
echo "-- latents_index line count --"
wc -l cfs_sd3_calib/latents/latents_index.csv 2>&1
echo "-- ENOSPC in sd3 logs --"
grep -c 'No space left' run_extract_sd3_gpu0.log run_extract_sd3.log 2>&1
echo "-- attempt count sd3 --"
grep -c 'attempt.*starting' run_extract_sd3_gpu0.log run_extract_sd3.log 2>&1
echo "-- gpu now --"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo "-- live workers --"
ps aux | grep -e run_extract -e activation_extractor | grep -v grep
echo "-- pixart npz --"
find cfs_pixart_calib/latents -name '*.npz' 2>/dev/null | wc -l
echo "-- pixart rows minus header --"
tail -n +2 cfs_pixart_calib/concept_metrics.csv | wc -l
echo "-- pixart ENOSPC/OOM --"
tail -n 25 run_extract_pixart.log
