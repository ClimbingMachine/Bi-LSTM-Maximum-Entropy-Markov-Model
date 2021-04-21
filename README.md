# Bi-LSTM-Max-Entropy-Markov-Random-Field-Model

Targeted sentiment analysis aims to capture sentiment expressed towards an entity in a given sentence. The task consists of two parts, Named Entity Recognition (identifying the entity) and identifying whether there is a sentiment directed towards the entity. For the purpose of this project, we will focus on doing them in a collapsed way (combining sentiment and named entity recognition into one label sequence, and trying to predict that sequence). 

This project implement Deep Maximum Entropy Markov Models (DMEMM) for the targeted sentiment task using the given dataset. Three different models/options are provided:
  1. Deep Maximum Entropy Markov Model with randomly initialized embedding.
  2. Deep Maximum Entropy Markov Model with pretrained Word2Vec embedding.
  3. Bi-LSTM Maximum Entropy Markov Model with randomly initialized/pretrained Word2Vec embeddings.
