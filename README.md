# STGCN-Tf2

## Spatio-Temporal Graph Convolutional Neural Network

This repository contains code updated for Tensorflow Version 2 (TF2 + Keras) from the original [repository](https://github.com/VeritasYin/STGCN_IJCAI-18). The Keras component is very limited. For TF2 only, check the branch named 'tf2-only'.

Other than code conversion from TFv1 to TFv2, certain additional changes have been made in the data loading mechanism, inference shapes not matching and a model update (Ko -= 2*(Kt - 1)). This code runs slower than the original version.

For my project, the most updated code is present in another [repository](https://github.com/Swadesh13/Pollution-STGCN) that has additional changes and also, a number of added models to compare various combinations of Spatial-GCN and Temporal-GCN layers.