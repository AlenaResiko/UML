Repo and Drive Layout
* Data & Preprocessing

Raw corpora sit under data/: music/ includes Taylor Swift lyric dumps grouped by album, scientific/ stores fetched arXiv sources/metadata, and huggingface/ is where datasets listed in data/huggingface/REGISTRY.json are materialized.

Programmatic ingestion is handled in uml_project/data/scientific/scientific.py (line 1) (download, tar extraction, title parsing, and JSON export of LaTeX sources) and /Users/uml_project/data/pre_processing/scentific_aboba.ipynb/

Actual sentence-level cleaning is implemented in uml_project/data/pre_processing/sentence.py and (line 1) (spaCy-based sentencizer with lyric-specific filtering, deduplication, and LaTeX-aware branches) and the supporting LaTeX scrubber in uml_project/data/pre_processing/latex_helper.py (line 1) (removes math/env blocks, inline commands, and normalizes prose before re-splitting). Self-supervised utilities such as synthetic label generation sit in uml_project/data/self_supervised/synthetic_labels.py (line 1).

* Model Training
Folder with notebooks: https://drive.google.com/drive/folders/1cwmpxFinmcvaEkfAmooxfWx6ZcE4gcgB
  * UML-Bert notebook is for training encoder+pooler via contrastive learning
  * UML-Eval notebook is for evaluating embedding metrics (uniforming, alignment, spearman rank, within-document similarity, t-SNE dataset cluster visualization)
  * UML-Plots notebook is for visualizing embedding metrics with respect to dimesionality
  * PCA is notebook for projecting datasets via PCA and evaluting performace of projections

* Evaluation & Metrics
https://colab.research.google.com/drive/1FUjgSg4oB_jOIPAKeXtG57VbYrQKayD2?usp=sharing

Results are in https://drive.google.com/drive/u/0/folders/1tlR0zWuS-A_NvsfKnnS6VFsQlovlsrnX

Running doc (paper analysis, suggested benchmarks + eval, random):
https://docs.google.com/document/d/1iEbmiGagz64P1kkLeR32FaTMiEfaba8-CwHGpI4GJw4/edit?usp=sharing


# Hyperparameters
train_test_split = 0.8
seed = 42
