## Outlier Resistant Unsupervised Deep Architectures for Attributed Network Embedding

To generate embeddings using model 1 (DONE) run following command
    python run_done.py --config config_file_path

This command reads configuration stored in json format from config_file_path and generates the embeddings. Sample configuration can be found at 'config_done'. Embeddings are stored in emb/ folder.

To generate embeddings using model 2 (AdONE) run following command
    python run_adone.py --config config_file_path

This command reads configuration stored in json format from config_file_path and generates the embeddings. Sample configuration  can be found at 'config_adone'. Embeddings are stored in emb/ folder.
