# FMFormer: Frame-padded Multi-scale Transformer for Monocular 3D Human Pose Estimation

Our paper has been published in IEEE Transactions on Multimedia: [Frame-Padded Multiscale Transformer for Monocular 3D Human Pose Estimation](https://ieeexplore.ieee.org/document/10374279).

This work is based on the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) and [MixSTE](https://github.com/JinluZhang1126/MixSTE), and you can get more help there.

<p align="center"> <img src="./figure/pose.gif" width="100%"> </p> 
<p align="center"> Test on Human3.6M </p>

## Environment

The code is conducted under the following environment:

* Ubuntu 20.04
* Python 3.9.16
* PyTorch 1.13.1
* CUDA 11.7

## Dataset

The dataset setting follow the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
Please refer to it to set up the Human3.6M dataset (under ./data directory).

# Evaluation

* Download our pretrained model from [Google Drive](https://drive.google.com/file/d/1ylUJ6VxMqulK_lAelzbXpw8RwtzgzXjz/view?usp=sharing);

Then run the command below (evaluate on 243 frames input):

```bash
python run.py -k cpn_ft_h36m_dbb -c <checkpoint_path> --evaluate <checkpoint_file> -f 243 -s 243 --edgepad 81
```

# Training from scratch

Training FMFormer with GPUs:

```bash
python run.py -k cpn_ft_h36m_dbb -f 243 -s 243 --edgepad 81 -l log/run -c checkpoint -gpu 0,1
```

## Acknowledgement

Thanks for the baselines, we construct the code based on them:

* VideoPose3D
* MixSTE

