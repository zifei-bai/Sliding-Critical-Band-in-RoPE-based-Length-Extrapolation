#!/bin/bash

set -e 


DATA_PATH="./data/"
SAVE_PATH="./models/"
RESULT_PATH="./results/"
GRAPH_PATH="./graphs/"
ORIGINAL_LEN=(50 100 500)


for ori in "${ORIGINAL_LEN[@]}"; do
    
    case $ori in
        50)
          
            ZW="{1.1:15, 1.2:15, 1.5:15, 2.0:15, 4.0:15, 8.0:15}" 
            D_EXTRA=30
            ;;
        100)
           
            ZW="{1.1:21, 1.2:17, 1.5:18, 2.0:17, 4.0:10, 8.0:6}"
            D_EXTRA=37
            ;;
        500)
           
            ZW="{1.1:10, 1.2:10, 1.5:10, 2.0:10, 4.0:10, 8.0:10}"
            D_EXTRA=53
            ;;
        *)
            
            echo "⚠️ 未知的 original_len: $ori"
            continue
            ;;
    esac

    
    python src/draw_graphs.py \
        --result_dir "${RESULT_PATH}" \
        --graph_dir "${GRAPH_PATH}" \
        --original ${ori} \
        --rope_base 10000 \
        --d_extra ${D_EXTRA} \
        --zoom_widths "${ZW}" \
        --ublb "ub"
        
    echo "Figure saved for original=${ori}"
done

python src/draw_graphs.py \
        --result_dir "${RESULT_PATH}" \
        --graph_dir "${GRAPH_PATH}" \
        --original 500 \
        --rope_base 100000 \
        --d_extra 43 \
        --zoom_widths "{1.1:25, 1.2:25, 1.5:25, 2:22, 4:24, 8:20}" \
        --ublb "ub"
        
echo "Figure saved for original=500, rb=100000"