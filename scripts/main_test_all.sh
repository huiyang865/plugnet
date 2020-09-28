array=(
  "IIIT5K_3000" 
  "svt_647"
  "ic03_867"
  "ic15_2077"
  "ic13_1015"
  "ic15_1811"
  "svt_p_645"
  "cute80_288")

DLS_DATA_URL="/home/yuming/yanghui/datasets/test_lmdb/"
DLS_TRAIN_URL=""
for i in "${array[@]}"
do
  echo $i
  CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --synthetic_train_data_dir xxx \
    --test_data_dir ${DLS_DATA_URL}$i \
    --batch_size 1024 \
    --workers 8 \
    --height 64 \
    --width 256 \
    --voc_type ALLCASES_SYMBOLS \
    --arch ResNet_ASTER \
    --with_lstm \
    --logs_dir logs/feature_enhanced \
    --real_logs_dir ./ \
    --max_len 100 \
    --evaluate \
    --STN_ON \
    --beam_width 5 \
    --tps_inputsize 32 64 \
    --tps_outputsize 32 100 \
    --tps_margins 0.05 0.05 \
    --stn_activation none \
    --num_control_points 20 \
    --resume /home/yuming/yanghui/aster_result/aster_sr_sharing_concate_0.01/logs/aster_sr_sharing_concate_0.01/model_best.pth.tar    
done



