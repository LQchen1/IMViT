# IMViT

This folder contains the implementation of the IMViT for image classification.

## Main Results on ImageNet-1K with Pretrained Models

| name | pretrain | resolution |acc@1 | acc@5 | #params | FLOPs | Throughput | 1K model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| IMViT-T | ImageNet-1K | 224x224 | 73.2 | 91.5 | 3.9M | 0.7G | 1680 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) |
| IMViT-S | ImageNet-1K | 224x224 | 79.8 | 95.0 | 9.8M | 1.8G | 1469 | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth) |
| IMViT-B | ImageNet-1K | 224x224 | 82.8 | 96.2 | 25.7M | 4.9G | 1177  | [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth) |

## Usage

### Install

We recommend using the pytorch docker `nvcr>=21.05` by
nvidia: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.

- Clone this repo:

```bash
git clone https://github.com/microsoft/Swin-Transformer.git
cd Swin-Transformer
```

- Create a conda virtual environment and activate it:

```bash
conda create -n swin python=3.7 -y
conda activate imvit
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

- Install `timm==0.4.12`:

```bash
pip install timm==0.4.12
```
- We use apex for mixed precision training by default. To install apex, run::

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```

- Install fused window process for acceleration, activated by passing `--fused_window_process` in the running script
```bash
cd kernels/window_process
python setup.py install #--user
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train_map.txt`, `val_map.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 data/ImageNet-Zip/val_map.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 data/ImageNet-Zip/train_map.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```


### Evaluation

To evaluate a pre-trained `IMViT` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```

### Training from scratch on ImageNet-1K

To train a `IMViT` on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```


For example, to train `IMViT` with 8 GPU on a single node for 300 epochs, run:

`IMViT-B`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs\IM_VIT\im_vit_base_224.yaml --data-path <imagenet-path> --batch-size 256 \
--accumulation-steps 2 [--use-checkpoint]
```

### Throughput

To measure the throughput, run:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size 128 --throughput --disable_amp
```
