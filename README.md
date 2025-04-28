## News
We are excited to share that the corresponding paper for this project has been accepted into the **SIGIR 2025 Proceedings**🎉🎉!

## Setup

To get started, follow these steps:

### 1. Install Required Packages
First, install all necessary dependencies by using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Prepare Netflix Dataset Node Features
Download the Netflix dataset node features and place them in the `netflix_data` directory. We recommend following the instructions provided by [LLMRec](https://github.com/HKUDS/LLMRec) to obtain the node features.

### 3. Run the Main Program
Once the environment is set up and the dataset is prepared, you can run the project by executing:

```bash
python main.py
```

## Detailed Datasets Info

We perform experiments on three publicly available datasets, *i.e.*, [Netflix](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data), [MovieLens](https://files.grouplens.org/datasets/movielens/ml-10m-README.html), and [Amazon-book](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).  
To ensure a fair comparison, we provide only textual information as side information for all methods.  
For baselines that cannot directly utilize textual information, such as LightGCN [[He et al., 2020]](#ref-he2020lightgcn), we use text encodings as node features.  
We follow existing work to split the train, validation, and test sets. For the Netflix and MovieLens datasets, we use the same split as in LLMRec [[Wei et al., 2024]](#ref-wei2024llmrec); for the Amazon-book dataset, we follow the split from RLMRec [[Ren et al., 2024]](#ref-ren2024representation).

- **Netflix dataset** [[Bennett et al., 2007]](#ref-bennett2007netflix)  
  The Netflix dataset is sourced from the Kaggle website.  
  We derive the interaction graph `G` and user interaction history `L_U` from the users' viewing history.  
  For items, we combine the movie title, year, genre, and categories as textual information, denoted as `T_V`, while for users, we use age, gender, favorite directors, country, and language as textual information, denoted as `P_U`.  
  Additionally, we use BERT [[Reimers and Gurevych, 2019]](#ref-reimers2019sentence) to encode the textual information of users and items, obtaining user features `F_U` and item features `M_V`, respectively.

- **MovieLens dataset** [[Harper and Konstan, 2015]](#ref-harper2015movielens)  
  The MovieLens dataset is sourced from ML-10M.  
  We obtain the interaction graph `G` and user interaction history `L_U` from the users' viewing history.  
  The side information, including movie title, year, and genre, is combined as the item textual feature `T_V`, while user age, gender, country, and language are combined as the user textual feature `P_U`.  
  Additionally, we encode these textual information using BERT [[Reimers and Gurevych, 2019]](#ref-reimers2019sentence) as features `F_U` and `M_V`.

- **Amazon-book dataset** [[Ni et al., 2019]](#ref-ni2019justifying)  
  The Amazon-book dataset contains book review records from 2000 to 2014.  
  We treat each review as a user-item interaction and derive the interaction graph `G` and user interaction history `L_U` from the records.  
  We use the book title, year, and categories as the textual information for items `T_V`, and the user's review content as the textual information for users `P_U`.  
  These information are encoded by BERT [[Reimers and Gurevych, 2019]](#ref-reimers2019sentence) to obtain features `F_U` and `M_V`.

## References
- <a name="ref-he2020lightgcn"></a>He, Xiangnan, et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." SIGIR 2020.
- <a name="ref-wei2024llmrec"></a>Wei, Xinyao, et al. "LLMRec: Towards Open-Ended Recommendation with Large Language Models." arXiv 2024.
- <a name="ref-ren2024representation"></a>Ren, Yujia, et al. "Representation Learning with Preference Graphs for Open-World Recommendation." WWW 2024.
- <a name="ref-bennett2007netflix"></a>Bennett, James, and Stan Lanning. "The Netflix Prize." KDD Cup 2007.
- <a name="ref-harper2015movielens"></a>Harper, F. Maxwell, and Joseph A. Konstan. "The MovieLens Datasets: History and Context." ACM TiiS 2015.
- <a name="ref-ni2019justifying"></a>Ni, Junnan, et al. "Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects." EMNLP 2019.
- <a name="ref-reimers2019sentence"></a>Reimers, Nils, and Iryna Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP/IJCNLP 2019.

## Pseudo code
We present the pseudo code for model training in Algorithm 1.
![Pseudo Code](/Pseudocode.png)
