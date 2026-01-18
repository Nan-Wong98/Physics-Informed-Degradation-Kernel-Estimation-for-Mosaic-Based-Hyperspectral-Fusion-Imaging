# Train for demosaicing
## CAVE
``` 
python train_simulate.py --idx 1 --data_path /data/wangn/DataSet --dataset CAVE --epochs 1000 --train_size 64 --stride 32 --batch_size 16 --lr_decay --save_freq 50 --device 1 --noise_level 0.001 0.005
```

## ICVL
```
python train_simulate.py --idx 2 --data_path /data/wangn/DataSet --dataset ICVL --epochs 10 --train_size 64 --stride 64 --batch_size 16 --lr_decay --save_freq 5 --device 2 --noise_level 0.001 0.005
```

## realworld dataset
```
python train_realworld.py --idx 3 --data_path /data/wangn/DataSet --dataset real_world --epochs 10 --train_size 64 --stride 64 --batch_size 16 --lr_decay --save_freq 1 --visual --visual_freq 1 --device 7
```

# Generate
## CAVE
```
# all
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_model ./1/model/best_166.pth --noise_level 0.001 0.005

# assign
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_model ./1/model/best_166.pth --noise_level 0.001 0.005 --data_id jelly_beans_ms.mat
```

## ICVL
```
# all
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset ICVL --load_model ./2/model/best_9.pth --noise_level 0.001 0.005

# assign
python generate.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset ICVL --load_model ./2/model/best_9.pth --noise_level 0.001 0.005 --data_id pepper_0503-1228.mat
```

## RealWorld
```
# all
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path ../../DataSet/ --dataset real_world --load_model ./3/model/best_1.pth

# assign
python generate.py --idx 1 --real_world --mosaic_save --pan_save --demosaic_save --data_path ../../DataSet/ --dataset real_world --load_model ./3/model/best_1.pth --data_id 1.raw 16.raw 37.raw
```

# test
## Cave
```
# test all
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/

# test assign
python test.py --idx 1 --simulate --data_path ./CAVE/1/result/mat/ --data_id paints_ms.mat
```
## ICVL
```
# test all
python test.py --idx 1 --simulate --data_path ./ICVL/1/result/mat/

# test assign
python test.py --idx 1 --simulate --data_path ./ICVL/1/result/mat/ --data_id Lehavim_0910-1630.mat
```
## RealWorld
```
# test all
python test.py --idx 1 --real_world --data_path ./real_world/1/result/mat/ --patch_size 64

# test assign
python test.py --idx 1 --real_world --data_path ./real_world/1/result/mat/ --patch_size 64 --data_id 3.mat
```

# Visualize
## simulate
```
# Visualize fused
# CAVE
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --save_path ./CAVE/1/result/rgb/fused/ --data_type fused --data_id jelly_beans_ms.mat --detach --detach_size 25 25 --detach_coordinate 210 98 210 98 --boxcolor b --boxwidth 1

# ICVL
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --save_path ./ICVL/1/result/rgb/fused/ --data_type fused --data_id pepper_0503-1228.mat --detach --detach_size 64 64 --detach_coordinate 135 625 135 625 --boxcolor b --boxwidth 3

# Visualize diffmap
# CAVE
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/fused/ --mat_path_for_diff ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/diffmap/ --data_id jelly_beans_ms.mat --detach --detach_size 25 25 --detach_coordinate 210 98 210 98 --boxcolor b --boxwidth 1 --mae_level 64 --sam_level 64

# ICVL
python visualize.py --visual_task diffmap --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/fused/ --mat_path_for_diff ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/diffmap/ --data_id pepper_0503-1228.mat --detach --detach_size 64 64 --detach_coordinate 135 625 135 625 --boxcolor b --boxwidth 3 --mae_level 16 --sam_level 16

# Visualize pan
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/pan/ --save_path ./CAVE/1/result/rgb/pan/ --data_type pan --data_id jelly_beans_ms.mat --detach --detach_size 25 25 --detach_coordinate 210 98 210 98 --boxcolor b --boxwidth 1
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/pan/ --save_path ./ICVL/1/result/rgb/pan/ --data_type pan --data_id Lehavim_0910-1630.mat --detach --detach_size 128 128 --detach_coordinate 300 600 100 160 --boxcolor r

# Visualize gt
# CAVE
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/gt/ --save_path ./CAVE/1/result/rgb/gt/ --data_type mosaic --data_id jelly_beans_ms.mat --detach --detach_size 25 25 --detach_coordinate 210 98 210 98 --boxcolor b --boxwidth 1

# ICVL
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/gt/ --save_path ./ICVL/1/result/rgb/gt/ --data_type mosaic --data_id pepper_0503-1228.mat --detach --detach_size 64 64 --detach_coordinate 135 625 135 625 --boxcolor b --boxwidth 3

# Visualize upmosaic
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./CAVE/1/result/mat/mosaic/ --save_path ./CAVE/1/result/rgb/upmosaic/ --data_type upmosaic --data_id jelly_beans_ms.mat --detach --detach_size 25 25 --detach_coordinate 210 98 210 98 --boxcolor b --boxwidth 1
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./ICVL/1/result/mat/mosaic/ --save_path ./ICVL/1/result/rgb/upmosaic/ --data_type upmosaic --data_id Lehavim_0910-1630.mat --detach --detach_size 128 128 --detach_coordinate 300 600 100 160 --boxcolor r
```

## real-world
```
# Visualizing fused detach
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 1.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 1780 270 1780 270 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 16.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 740 270 740 270 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/fused/ --save_path ./real_world/1/result/rgb/fused/ --data_id 37.mat --data_type fused --detach --detach_size 128 128 --detach_coordinate 740 770 740 770 --boxwidth 5 --boxcolor b

# Visualize pan real-world detach
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 1.mat --data_type pan --detach --detach_size 128 128 --detach_coordinate 1780 270 1780 270 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 16.mat --data_type pan --detach --detach_size 128 128 --detach_coordinate 740 270 740 270 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/pan/ --save_path ./real_world/1/result/rgb/pan/ --data_id 37.mat --data_type pan --detach --detach_size 128 128 --detach_coordinate 740 770 740 770 --boxwidth 5 --boxcolor b

# Visualize upmosaic real-world detach
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 1.mat --data_type upmosaic --detach --detach_size 128 128 --detach_coordinate 1780 270 1780 270 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 16.mat --data_type upmosaic --detach --detach_size 128 128 --detach_coordinate 740 270 740 270 --boxwidth 5 --boxcolor b
python visualize.py --visual_task rgb --spatial_ratio 8 --mat_path ./real_world/1/result/mat/mosaic/ --save_path ./real_world/1/result/rgb/upmosaic/ --data_id 37.mat --data_type upmosaic --detach_size 128 128 --detach --detach_coordinate 740 770 740 770 --boxwidth 5 --boxcolor b
```

## real-world video
```
# Visualizing fused detach
python visualize.py --visual_task rgb_video --spatial_ratio 8 --mat_path ./real_world_video/1/result/mat/fused/ --save_path ./real_world_video/1/result/rgb/fused/ --data_id 11_160.mat --data_type fused --detach --detach_size 380 580 --detach_coordinate 810 830 810 830 --boxwidth 5 --boxcolor b

# Visualize pan real-world detach
python visualize.py --visual_task rgb_video --spatial_ratio 8 --mat_path ./real_world_video/1/result/mat/pan/ --save_path ./real_world_video/1/result/rgb/pan/ --data_id 11_160.mat --data_type pan --detach --detach_size 380 580 --detach_coordinate 810 830 810 830 --boxwidth 5 --boxcolor b

# Visualize upmosaic real-world detach
python visualize.py --visual_task rgb_video --spatial_ratio 8 --mat_path ./real_world_video/1/result/mat/mosaic/ --save_path ./real_world_video/1/result/rgb/upmosaic/ --data_id 11_160.mat --data_type upmosaic --detach --detach_size 380 580 --detach_coordinate 810 830 810 830 --boxwidth 5 --boxcolor b
```

## latent variable
```
# CAVE
python .\code_for_fig8.py --idx 1 --simulate --mosaic_save --pan_save --demosaic_save --gt_save --data_path ../../DataSet/ --dataset CAVE --load_model ./1/model/best_38.pth --noise_level 0.01 0.05 --data_id jelly_beans_ms.mat  --detach_size 25 25 --detach_coordinate 210 98 210 98 --boxcolor b --boxwidth 1

# real-world
python .\code_for_fig8.py --idx 1 --real_world --data_path ../../DataSet/ --dataset real_world --load_model ./3/model/best_4.pth --data_id 1.raw  --detach_size 128 128 --detach_coordinate 1780 270 1780 270 --boxcolor b --boxwidth 5
```

## degrade visualzie
```
python .\code_for_fig9.py --idx 1 --data_path ../../DataSet/ --dataset real_world --load_model ./3/model/best_4.pth --data_id 1.raw
```