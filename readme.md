# Official implementation of Representation Learning via Consistent Assignment of Views over Random Partitions (CARP)
**Thirty-seventh Conference on Neural Information Processing Systems**

## Important Links

- [Project website](https://sthalles.github.io/carp/)
- [arXiv](https://arxiv.org/abs/2310.12692)
- [NeurIPS procedings](https://openreview.net/forum?id=fem6BIJkdv&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2023%2FConference%2FAuthors%23your-submissions))

## Download pre-trained models

Make sure to download CARP's pretraining files and place them in ```/pretrained/carp/``` folder.

[Checkpoints can be downloaded here.](https://drive.google.com/drive/folders/12zKF5L55kS0oNhHNTwKbCRUkAh-P0zlv?usp=sharing)

## Performance

|Method|Epochs|Multicrop|Top-1|k-NN|URL|
|-|-|-|-|-|-|
|CARP|100|2x224 + 6x96|72.5|63.5|[CARP-100ep](https://drive.google.com/drive/folders/1Kj7pp2CcUcEoLYv2d4vs8hQK6xTK6VbR?usp=sharing)|
|CARP|200|2x224 + 6x96|74.2|66.5|[CARP-100ep](https://drive.google.com/drive/folders/1NmEAzD4BtM33rOgjEw3o8YS9vHj7qrH9?usp=sharing)|
|CARP|400|2x224|73.0|67.6|[CARP-400ep](https://drive.google.com/drive/folders/1xlDsn0JsD_tB11HA1qjdDJxVVy85pV2z?usp=sharing)|
|CARP|400|2x224 + 6x96|75.3|67.7|[CARP-400ep](https://drive.google.com/drive/folders/1feMX0I7u_mIafiYP4EafJforZXHRkvBX?usp=sharing)|

## Running evaluations

To run evaluations, ensure you have a proper Python environment with PyTorch 2.0 and other dependencies. 

Go to specific evaluation folders (such as knn or kmeans) for examples of how to run each.

# Reference

Please, cite this work as:

```
@inproceedings{
  Silva2023,
  title={Representation Learning via Consistent Assignment of Views over Random Partitions},
  author={Silva, Thalles and Ram\'irez Rivera, Ad\'in},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems ({NeurIPS})},
  year={2023},
  url={https://openreview.net/forum?id=fem6BIJkdv}
}
```
