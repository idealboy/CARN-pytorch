#rm -rf ./DIV2K/*
python carn/sample.py --model carn --test_data_dir dataset/DIV2K --scale 2 --N 4 --ckpt_path ./checkpoint/carn.pth --sample_dir ./
