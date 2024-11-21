# BEACON

This repository provides the code and data for the paper

> Bridging chemical structure and conceptual knowledge enables accurate prediction of compound-protein interaction

## Citation
If you want to use our code and datasets in your research, please cite:

```
@article{tao2024bridging,
  title={Bridging chemical structure and conceptual knowledge enables accurate prediction of compound-protein interaction},
  author={Tao, Wen and Lin, Xuan and Liu, Yuansheng and Zeng, Li and Ma, Tengfei and Cheng, Ning and Jiang, Jing and Zeng, Xiangxiang and Yuan, Sisi},
  journal={BMC biology},
  volume={22},
  number={1},
  pages={248},
  year={2024},
  publisher={Springer}
}
```

## 1. Requirements

To run the code, you need the following dependencies:

```
pytorch                       1.5.0
rdkit                         2018.09.3
```

## 2. Data Download

The data used in this experiment can be downloaded from Google Drive:

```
https://drive.google.com/drive/folders/1kXIjwf9BmTCb9GZzukIOkInJoyoftGLa?usp=sharing
```

## 3. Training and Evaluation

Please use the following command:

```bash
python main.py --dataset human --iteration 3
```

## 4. Using Custom Datasets

### Get Protein Features on Your Dataset

```bash
python protein_feature_extraction.py --input_file drugbank_protein_sequences.csv --output_file drugbank_protein_features.npy
```

### Get KG Embeddings on Your Dataset

To install the latest version of DGL-KE run:

```bash
sudo pip3 install dgl
sudo pip3 install dglke
```

Train a `transE` model on `DRKG` dataset by running the following command:

```bash
dglke_train --dataset DRKG --data_path ./ --data_files drkg.tsv drkg.tsv drkg.tsv \
--format 'raw_udd_hrt' --model_name TransE_l2 --batch_size 2048 --neg_sample_size 256 --hidden_dim 400 --gamma 12.0 --lr 0.1 --max_step 100000 \
--log_interval 1000 --batch_size_eval 16 -adv --regularization_coef 1.00E-07 --test --num_thread 7 --gpu 0 --neg_sample_size_eval 10000
```

This command will use the `DRKG` dataset, train the `transE` model and save the trained embeddings into the file.

For any clarification, comments, or suggestions, please create an issue or contact [Wen Tao](taowen@hnu.edu.cn).
