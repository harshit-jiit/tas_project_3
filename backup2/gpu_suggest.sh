#!/bin/bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | awk -F, '
BEGIN {
    print "GPU Recommendations:"
    print "==================="
    available_gpus = ""
}
{
    gpu_id = $1
    mem_used = $2
    mem_total = $3
    util = $4
    
    # Consider GPU available if memory usage < 1000MB or utilization < 10%
    if (mem_used < 1000 && util < 10) {
        if (available_gpus == "") {
            available_gpus = gpu_id
        } else {
            available_gpus = available_gpus "," gpu_id
        }
        print "GPU " gpu_id ": Available (Memory: " mem_used "MB/" mem_total "MB, Util: " util "%)"
    } else {
        print "GPU " gpu_id ": In Use (Memory: " mem_used "MB/" mem_total "MB, Util: " util "%)"
    }
}
END {
    print ""
    if (available_gpus != "") {
        print "Recommended GPUs to use: " available_gpus
        split(available_gpus, arr, ",")
    } else {
        print "No GPUs currently available"
    }
}'