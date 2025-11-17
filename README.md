# PUTFormer
This is the official pytorch implementation of "Phase unwrapping via fully exploiting global and local spatial dependencies" published in Optics & Laser Technology 2025.

## Requirements
torch>=1.10

## Data
Synthetic Data: Follows [SQDLSTM](https://github.com/Laknath1996/DeepPhaseUnwrap). For synthetic data, it is better to generate it on your own and retrain all baselines for fair comparison.
FPP Data: Follows [HiPhase](https://github.com/WanzhongSong/HiPhase)

The data should be preprocessed to h5py files as in [SQDLSTM](https://github.com/Laknath1996/DeepPhaseUnwrap).