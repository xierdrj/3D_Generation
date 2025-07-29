'''
python -c "import numpy as np;print(np.load('1a04e3eab45ca15dd86060f189eb133.npy').shape)"

python main.py --base configs/base/airplane.yaml -t True --gpus 0, -n airplane_base_train

python main.py --base configs/base/bag.yaml -t True --gpus 0,1,2,3 -n bag_base_train

python main.py --base configs/stage1/128/bag.yaml -t True --gpus 0,1,2,3 -n bag_stage1_train

python main.py --base configs/stage2/128/bag.yaml -t True --gpus 0,1,2,3 -n bag_stage2_train

python main.py --base configs/stage3/128/bag.yaml -t True --gpus 0,1,2,3 -n bag_stage3_train


python main.py --base configs/base/our_data.yaml -t True --gpus 0,1,2,3 -n our_data_base_train

python main.py --base configs/stage1/128/our_data.yaml -t True --gpus 0,1,2,3 -n our_data_stage1_train

python main.py --base configs/stage2/128/our_data.yaml -t True --gpus 0,1,2,3 -n our_data_stage2_train

python main.py --base configs/stage3/128/our_data.yaml -t True --gpus 0,1,2,3 -n our_data_stage3_train
测试
'''