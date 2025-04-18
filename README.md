# Texture-Aware Self-Attention Model for Hyperspectral Tree Species Classification

[Nanying Li](https://scholar.google.com.hk/citations?user=NwzUe2YAAAAJ&hl=zh-CN&oi=ao), [Shuguo Jiang](https://scholar.google.com.hk/citations?hl=zh-CN&user=B1YTGUgAAAAJ), [Jiaqi Xue](), [Songxin Ye](), [Sen Jia](https://scholar.google.com.hk/citations?hl=zh-CN&user=UxbDMKoAAAAJ)

<hr />

<div style="text-align: justify;">
> **Abstract:** *Forests play an irreplaceable role in carbon sinks. However, there are obvious differences in the carbon sink capacity of different tree species, so the scientific and accurate identification of surface forest vegetation is the key to achieving the double carbon goal. Due to the disordered distribution of trees, varied crown geometry, and high difficulty in labeling tree species, traditional methods have a poor ability to represent complex spatial–spectral structures. Therefore, how to quickly and accurately obtain key and subtle features of tree species to finely identify tree species is an urgent problem to be solved in current research. To address these issues, a texture-aware self-attention model (TASAM) is proposed to improve spatial contrast and overcome spectral variance, achieving accurate classification of tree species hyperspectral images (HSIs). In our model, a nested spatial pyramid module is first constructed to accurately extract the multiview and multiscale features that highlight the distinction between tree species and surrounding backgrounds. In addition, a cross-spectral–spatial attention module is designed, which can capture spatial–spectral joint features over the entire image domain. The Gabor feature is introduced as an auxiliary function to guide self-attention to autonomously focus on latent space texture features, further extract more appropriate and accurate information, and enhance the distinction between the target and the background. Verification experiments on three tree species hyperspectral datasets prove that the proposed method can obtain finer and more accurate tree species classification under the condition of limited labeled samples. This method can effectively solve the problem of tree species classification in complex forest structures and can meet the application requirements of tree species diversity monitoring, forestry resource investigation, and forestry carbon sink analysis based on HSIs.* 
</div>
<hr />


## 1. Create Envirement:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## 2. Data Preparation:
- Pre-extracting 3D Gabor features of tree specifies from hyperspectral images.

  ```shell
  python gabor.py -i (data path: 数据路径) -o (saving path: 保存目录)
  ```

## 3. Training

To train a model, run

```shell
python main.py --name (data name: 数据集名称)
```

## 4. Prediction:

To test a trained model, run 

```shell
python predict.py --name (data name: 数据集名称)
```


## Citation
If this repo helps you, please consider citing our works:


```
@article{li2023texture,
  title={Texture-aware self-attention model for hyperspectral tree species classification},
  author={Li, Nanying and Jiang, Shuguo and Xue, Jiaqi and Ye, Songxin and Jia, Sen},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--15},
  year={2023},
}
```
