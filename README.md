# PUTFormer
This is the official pytorch implementation of "Phase unwrapping via fully exploiting global and local spatial dependencies" published in Optics & Laser Technology 2025.

## Requirements
torch>=1.10

## Data
Synthetic Data: Follows [SQDLSTM](https://github.com/Laknath1996/DeepPhaseUnwrap). For synthetic data, it is better to generate it on your own and retrain all baselines for fair comparison.

FPP Data: Follows [HiPhase](https://github.com/WanzhongSong/HiPhase)

The data should be preprocessed to h5py files as in [SQDLSTM](https://github.com/Laknath1996/DeepPhaseUnwrap).

## Note
The LTBs implementation is upgraded to factorized attention inspired by [SERT](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Spectral_Enhanced_Rectangle_Transformer_for_Hyperspectral_Image_Denoising_CVPR_2023_paper.pdf). Instead of concatenation, we use sequential stacking.

## Citation
@article{quan2025phase,
  title={Phase unwrapping via fully exploiting global and local spatial dependencies},
  author={Quan, Yuhui and Yao, Xin and Chen, Zhifeng and Ji, Hui},
  journal={Optics \& Laser Technology},
  volume={181},
  pages={111872},
  year={2025},
  publisher={Elsevier}
}