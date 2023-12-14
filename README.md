# Machine-learning Analysis of Opioid Use Disorder Informed by MOR, DOR, KOR, NOR and ZOR-Based Interactome Networks

---
This script is for the paper "Multiscale differential geometry learning of networks with applications to single-cell RNA sequencing data, Hongsong Feng, Sean Cottrell, Yuta Hozumi, and  Guo-Wei Wei".

## Requirements

Python Dependencies
- python (>=3.7)
- numpy (1.17.4)
- scikit-learn (0.23.2)
- scipy (1.5.2)
- pandas (0.25.3)


## Download the repository
Download the repository from Github
```shell
# download repository by git
git clone https://github.com/WeilabMSU/MDG.git
```


Download and install the pretrained model under the downloaded OUD_PPI folder.

```shell
cd MDG
wget https://weilab.math.msu.edu/Downloads/MDG/features-CCP-UMAP.zip  
wget https://weilab.math.msu.edu/Downloads/MDG/scRNA-seq-data.zip  
unzip features-CCP-UMAP.zip  
unzip scRNA-seq-data.zip  
```

## Generating differential geometry features for MDG modeling.

```python
# use dataset GSE45719 for demonstration and kappa is set to 5 and 10 in our paper.
cd MDG
python mdg-curvature.py --dataset_name GSE45719 --kappa 5
```
The generated features are saved in the folder "features-CCP-UMAP".


## Reference

1. Multiscale differential geometry learning of networks with applications to single-cell RNA sequencing data, Hongsong Feng, Sean Cottrell, Yuta Hozumi, and  Guo-Wei Wei in print (2023).


## License
All codes released in this study is under the MIT License.
