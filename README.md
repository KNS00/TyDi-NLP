# Performing language models on the English TyDi QA dataset 

In this project, various NLP tasks are conducted on the Answerable TyDi QA dataset [1]. It was originally a multilingual project, created as a part of a NLP course at DIKU, where the models were applied on the Arabic, Bengali and Indonesian languages. Parts of the original project have been excluded for academic integrity.

## Getting Started

The project has been containerized for reproducibility. You should be able to run it with Docker.

### Prerequisites 

Download and open [Docker Desktop](https://www.docker.com/products/docker-desktop/) on your machine.

### Installation

To set up an environment and run the notebooks consistently:

```
docker build -t tydi_qa .
```

and run it by the following command
```
docker-compose up
```
A link to the notebooks should appear in the terminal. Open in your favourite browser to run them.


### Reproducing the resuls

Each notebook corresponds to a section in the original problem description.

Section 1 provides basic statistics on the Tydi QA data set. Among other stats, the table below is generated:
| Dataset        | Questions | Avg. Words/Question | Answerable Ratio | Total Words | Unique Words |
|----------------|---------------------:|-------------------------:|-----------------:|--------------------------------:|---------------------:|
| Train Set      |                7389 |                    8.06 |              0.5 |                          59533 |                 5018 |
| Validation Set |                 990 |                    8.20 |              0.5 |                           8122 |                 1230 |




Section 2 implements two language models: 

| Model   | Perplexity |
|---------|------------|
| Unigram | 720        |
| Bigram  | 49         |

As seen in the notebook, the unigram model also outputs a distribution on the tokens (words) in the data.

Section 3 implements three binary classifiers to predict whether a question is answerable or not:
| Model  | Accuracy |
| ----- | ----- |
| Logistic Regerssion with BPEmb embeddings  | 67%  |
| Logistic Regerssion with GloVe embeddings  | 65%  |
| BERT transformer neural network | 87% |

* Sections 4, 5 and 6 are not part of the public repository.

[1] Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. 2020. Tydi qa: A benchmark for information-seeking question answering in typologically diverse languages. Transactions of the Association for Computational Linguistics. Link: https://huggingface.co/datasets/copenlu/answerable_tydiqa.
