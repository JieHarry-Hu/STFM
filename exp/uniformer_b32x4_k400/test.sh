work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
python tools/run_net.py \
  --cfg $work_path/test.yaml \
  DATA.PATH_TO_DATA_DIR ./data_list/k400/ \
  DATA.PATH_PREFIX ./data_list/k400/ \
  DATA.PATH_LABEL_SEPARATOR "," \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.BATCH_SIZE 16 \
  NUM_GPUS 1 \
  UNIFORMER.DROP_DEPTH_RATE 0.0 \
  SOLVER.MAX_EPOCH 110 \
  SOLVER.BASE_LR 2e-4 \
  SOLVER.WARMUP_EPOCHS 10.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 1 \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH ./exp/uniformer_b32x4_k400/checkpoints/checkpoint.pyth \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path