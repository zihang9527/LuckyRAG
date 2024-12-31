python3 /cloudide/workspace/LuckyRAG/eval.py \
        --input_file "./data/RGB/zh_refine.json" \
        --persist_file_path './data/db/chromadb.db' \
        --collection_name 'zh_refine' \
        --api_key '' \
        --passage_num 5 \
        --noise_rate 0.6 \
        --top_n 3
        