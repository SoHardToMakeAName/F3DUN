# F3DUN
Official implementation for paper *Rethinking 3D-CNN in Hyperspectral Image Super-Resolution*. We present a full 3D CNN-based model, F3DUN, for hyperspectral image super-resolution, and achieve good performance on the CAVE and Harvard dataset.
[paper](https://www.mdpi.com/2072-4292/15/10/2574)
![The proposed Full 3D U-Net](https://github.com/SoHardToMakeAName/F3DUN/blob/main/figs/arch.PNG)
![Results on the CAVE dataset](https://github.com/SoHardToMakeAName/F3DUN/blob/main/figs/cave.PNG)
![Results on the Harvard dataset](https://github.com/SoHardToMakeAName/F3DUN/blob/main/figs/harvard.PNG)
## How to train your model
1. Download the [CAVE dataset](https://www.cs.columbia.edu/CAVE/databases/multispectral) and the [Harvard dataset](http://vision.seas.harvard.edu/hyperspec/index.html) to a anywhere you like.
2. Run *prepare_cave.py* or *prepare_Harvard.py* to generate data for training, validation and test. Remeber to place the generated data to a new folder called *datasets* in the same folder of *mains.py*.
3. Run the command below:
```
python -u mains.py train --name "F3DN" --model_config "configs/F3DN.yml" --n_blocks 4 --dataset_name CaveL --n_scale 4 --seed 3000 --plateau
python -u mains.py train --name "F3DN" --model_config "configs/F3DN.yml" --n_blocks 4 --dataset_name Harvard --n_scale 4 --seed 3000 --plateau
```
If you find this repository helpful, please cite our paper.
```
@Article{rs15102574,
AUTHOR = {Liu, Ziqian and Wang, Wenbing and Ma, Qing and Liu, Xianming and Jiang, Junjun},
TITLE = {Rethinking 3D-CNN in Hyperspectral Image Super-Resolution},
JOURNAL = {Remote Sensing},
VOLUME = {15},
YEAR = {2023},
NUMBER = {10},
ARTICLE-NUMBER = {2574},
URL = {https://www.mdpi.com/2072-4292/15/10/2574},
ISSN = {2072-4292},
DOI = {10.3390/rs15102574}
}
```
