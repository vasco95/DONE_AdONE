## Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding
This repository contains code for the WSDM 2020 paper *[Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding](https://dl.acm.org/doi/10.1145/3336191.3371788)*

To generate embeddings using model 1 *(DONE)* run following command
    python run_done.py --config config_file_path

This command reads configuration stored in json format from config_file_path and generates the embeddings. Sample configuration can be found at 'config_done'. Embeddings are stored in emb/ folder.

To generate embeddings using model 2 *(AdONE)* run following command
    python run_adone.py --config config_file_path

This command reads configuration stored in json format from config_file_path and generates the embeddings. Sample configuration  can be found at 'config_adone'. Embeddings are stored in emb/ folder.

You can cite the paper at
```
@inproceedings{10.1145/3336191.3371788,
author = {Bandyopadhyay, Sambaran and N, Lokesh and Vivek, Saley Vishal and Murty, M. N.},
title = {Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding},
year = {2020},
isbn = {9781450368223},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3336191.3371788},
doi = {10.1145/3336191.3371788},
booktitle = {Proceedings of the 13th International Conference on Web Search and Data Mining},
pages = {25–33},
numpages = {9},
keywords = {social networks, network representation learning, adversarial learning, community outliers, graph mining, deep autoencoder},
location = {Houston, TX, USA},
series = {WSDM ’20}
}
```
