work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --cfg $work_path/config.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/sthv2 \
  DATA.PATH_PREFIX ./data_list/sthv2/extracted_frames \
  DATA.LABEL_PATH_TEMPLATE "somesomev2_rgb_{}_split.txt" \
  DATA.IMAGE_TEMPLATE "{:06d}.jpg" \
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 2 \
  TRAIN.CHECKPOINT_PERIOD 2 \
  TRAIN.BATCH_SIZE 8 \
  NUM_GPUS 1 \
  UNIFORMER.DROP_DEPTH_RATE 0.4 \
  SOLVER.MAX_EPOCH 110 \
  SOLVER.BASE_LR 5e-4 \
  SOLVER.WARMUP_EPOCHS 5.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 3 \
  OUTPUT_DIR $work_path
