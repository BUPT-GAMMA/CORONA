# CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models

## News
- Our paper has been accepted to **SIGIR 2025** 🎉

## Introduction
CORONA is a coarse-to-fine recommendation framework that retrieves user-neighborhoods on user–item bipartite graphs and leverages LLM-augmented user profiles as side information. The coarse stage retrieves candidate users via graph-aware similarity with distance priors; the fine stage constructs compact subgraphs for downstream training/evaluation.

## Repository Structure
- `main.py`: training/validation/testing for user retriever
- `model.py`: retriever model with distance-aware embedding transformation
- `construct_graph.py`: build per-user subgraphs from retrieved neighbors
- `chat_api_query.py`: LLM-based user profiling and embedding generation
- `load.py`: utilities for dataset loading and diagnostics
- `netflix_data/`: example dataset placeholder (train/val/test splits and sparse matrices)

## Quick Start
### 1) Environment
- Python 3.9+
- Install minimal deps:
```bash
pip install -r requirements-min.txt
```
For CUDA/Torch Geometric GPU wheels, follow the official guides.

### 2) Configure
Create `.env` (or copy `.env.example`) to specify paths/devices:
```bash
cp .env.example .env
```
Key variables:
- `DATA_DIR`: project root for data and outputs (default `.`)
- `DATASET_DIR`: dataset subdir (default `/netflix_data`)
- `CUDA_VISIBLE_DEVICES`: GPU id (default `0`)
- `TOP_K`: retrieved users per query (default `500`)
- `OPENAI_*`: LLM credentials for profiling

### 3) Prepare Dataset
We experiment on Netflix, MovieLens, and Amazon-Book. Provide only textual side information for all methods.
- Place processed files under `${DATA_DIR}${DATASET_DIR}`:
  - `train.json`, `val.json`, `test.json` (uid -> item list)
  - `train_mat` (scipy sparse user–item CSR, pickled)
  - `augmented_user_init_embedding_final` (numpy array pickled, dim = user embedding)
  - Optional: `netflix_image_text/item_attribute.csv` for profiling
- For Netflix node features, we recommend following LLMRec instructions.

### 4) LLM-based User Profiling (Optional)
If you need to generate `augmented_user_init_embedding_final`:
```bash
make augment
```
This reads `train_mat`/`test.json` and writes `${AUGMENT_FILE_PATH}/augmented_user_init_embedding_final`.

### 5) Train and Evaluate
```bash
make train
```
After training, the best model and retrieved nodes are saved to `${DATA_DIR}/Graph_RA_Rec/model_states/`.
For testing independently:
```bash
make test
```

### 6) Construct Subgraphs
```bash
make graphs
```
Produces user/item subgraphs under `${DATA_DIR}/Graph_RA_Rec/${basename(DATASET_DIR)}/`.

## Datasets Details
- Netflix (KDD Cup 2007)
- MovieLens-10M (ACM TiiS 2015)
- Amazon-Book (EMNLP 2019)
We follow LLMRec for Netflix/MovieLens splits and RLMRec for Amazon-Book. Textual info is encoded by Sentence-BERT.

## Reproducibility Notes
- Determinism: `set_seed(3)` in `main.py`
- GPU selection via `CUDA_VISIBLE_DEVICES`
- Cached tensors: `*_for_RA.pkl` are stored in `${DATA_DIR}${DATASET_DIR}`

## Pseudocode
![Pseudo Code](/Pseudocode.png)

## Citation
If you find this repository helpful, please cite:
```
@inproceedings{corona2025,
  title={CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models},
  booktitle={Proceedings of the ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2025}
}
```

## References
- He et al., LightGCN, SIGIR 2020
- Wei et al., LLMRec, arXiv 2024
- Ren et al., RLMRec, WWW 2024
- Bennett and Lanning, The Netflix Prize, KDD Cup 2007
- Harper and Konstan, MovieLens, ACM TiiS 2015
- Reimers and Gurevych, Sentence-BERT, EMNLP/IJCNLP 2019

## License
This code is released for research purposes. See repository license if provided.
