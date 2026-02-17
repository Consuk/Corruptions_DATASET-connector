CUDA_VISIBLE_DEVICES=6 python evaluate_depth.py \
  --data_path /workspace/datasets/hamlyn/Hamlyn \
  --eval_split hamlyn \
  --dataset hamlyn \
  --load_weights_folder /workspace/repos/AF-SfMLearner/hamlyn_asfm/models/weights_19/ \
  --eval_mono \
  --hamlyn_strict_neighbors \
  --eval_filelist /workspace/datasets/hamlyn/splits/test_files2.txt \
  --gt_depths_path /workspace/datasets/hamlyn/splits/test_files2_gt_depths.npz \
  --min_depth 1 \
  --max_depth 300


  CUDA_VISIBLE_DEVICES=6 python eval_endovis_corruptions.py \
  --dataset_type hamlyn \
  --corruptions_root /workspace/datasets/hamlyn/hamlyn_corruptions_test \
  --load_weights_folder /workspace/repos/AF-SfMLearner/hamlyn_asfm/models/weights_19/ \
  --split_file /workspace/datasets/hamlyn/splits/test_files2.txt \
  --gt_depths_file /workspace/datasets/hamlyn/splits/test_files2_gt_depths.npz \
  --height 256 --width 320 --batch_size 16 \
  --output_csv hamlyn_corruptions_summary.csv


python csv_avg.py

CUDA_VISIBLE_DEVICES=6 python eval_endovis_corruptions.py   
--dataset_type hamlyn   
--corruptions_root /workspace/datasets/hamlyn/hamlyn_corruptions_test   
--load_weights_folder /workspace/repos/AF-SfMLearner/hamlyn_asfm/models/weights_19/    
--split_file /workspace/datasets/hamlyn/splits/test_files2.txt   
--gt_depths_file /workspace/datasets/hamlyn/splits/test_files2_gt_depths.npz   
--min_depth 1e-3 
--max_depth 300   
--strict   
--output_csv hamlyn_corruptions_summary_AF.csv

