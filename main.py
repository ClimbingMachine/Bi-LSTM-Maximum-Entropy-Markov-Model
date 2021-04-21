#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import string
import argparse
import csv
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def read_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {}
            sent, tag_string = line.strip().split('####')
            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # word sequence
            words = []
            # word caps
            caps = []
            # start word or not
            start = []
            flag = 1
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                if word not in string.punctuation:
                    # is capitalized or not
                    caps.append(word[0].isupper())
                    # lowercase the words
                    words.append(word.lower())
                else:
                    # is capitalized or not
                    caps.append(word.isupper())
                    # replace punctuations with a special token
                    words.append('PUNCT')
                
                if flag == 1:
                    start.append(flag)
                    flag = 0
                else:
                    start.append(flag)
                    
                if tag == 'O':
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            record['ts_raw_tags'] = ts_tags.copy()
            record['caps'] = caps.copy()
            record['start'] = start.copy()
            dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset

def SaveFile(given_df, output_csv_file):

    with open(output_csv_file, mode='w', encoding="utf-8") as out_csv:
        writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writerow(["sentence", "words", "ts_raw_tags", "ts_pred_tags"])
        for index, row in given_df.iterrows():
            writer.writerow([row['sentence'], row['words'], row['ts_raw_tags'], row['ts_pred_tags']])


# In[8]:


# model for option 1
class NN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, tag_size):
        super(NN, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 32)
        self.linear2 = nn.Linear(32, tag_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs


# In[19]:


# model for option 2
class NeuralN(nn.Module):

    def __init__(self, embedding_dim, vocab_size, context_size, tag_size):
        super(NeuralN, self).__init__()
        self.feature_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1    = nn.Linear(context_size*embedding_dim + 300, 32)      
        self.linear2    = nn.Linear(32, tag_size)

    def forward(self, inputs, char_sequence):
        embeds = inputs.view((1, -1))
        char_embeds = self.feature_embeddings(char_sequence).view((1, -1))
        cat = torch.cat((embeds, char_embeds), dim=1)
        out = F.relu(self.linear1(cat))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim = 1)
        return log_probs


# In[20]:


# Bi-LSTM model for model 3

class BiLSTM(nn.Module):

    def __init__(self, word_embedding_dim, feat_embedding_dim, word_hidden_dim,
                 feat_hidden_dim, word_vocab_size, feat_vocab_size, tagset_size):

        super(BiLSTM, self).__init__()
        self.word_hidden_dim = word_hidden_dim
        self.feat_hidden_dim = feat_hidden_dim

        self.word_embeddings = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.feat_embeddings = nn.Embedding(feat_vocab_size, feat_embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.feat_lstm = nn.LSTM(feat_embedding_dim, feat_hidden_dim//2, bidirectional = True)
        self.word_lstm = nn.LSTM(word_embedding_dim + feat_hidden_dim*4, word_hidden_dim//2, bidirectional = True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(word_hidden_dim, tagset_size)
        self.word_hidden = self.init_hidden(self.word_hidden_dim)
        self.feat_hidden = self.init_hidden(self.feat_hidden_dim)

    def init_hidden(self, size, condition = False):
        # Before we've done anything, we dont have any hidden state.
        return (torch.zeros(2, 1, size//2),
                torch.zeros(2, 1, size//2))

    def forward(self, word_sequence, feat_sequence):
        word_embeds = self.word_embeddings(word_sequence)

        feat_embeds = self.feat_embeddings(feat_sequence)
        feat_lstm_out, self.feat_hidden = self.feat_lstm(feat_embeds.view(len(feat_sequence), 1, -1), self.feat_hidden)
        
        x1 = word_embeds.view(len(word_sequence),1,-1)
        x2 = feat_lstm_out.view(len(word_sequence),1,-1)

        concat = torch.cat((x1, x2), dim=2)
        word_lstm_out, self.word_hidden = self.word_lstm(concat, self.word_hidden)

        tag_space = self.hidden2tag(word_lstm_out.view(len(word_sequence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# In[31]:


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/twitter1_train.txt', help='Train file')
    parser.add_argument('--test_file', type=str, default='data/twitter1_test.txt', help='Test file')
    parser.add_argument('--test_predictions_file', type=str, default='data/test_predictions.csv', help='Test file')
    parser.add_argument('--option', type=int, default=1, help='Option to run (1 = Randomly Initialized, 2 = Word2Vec, 3 = Bi-LSTM')
    args = parser.parse_args()
    
    # read the dataset
    train_set = read_data(path=args.train_file)
    # pre-process the data to add additional features
    train_set = pd.DataFrame.from_dict(train_set)
    train_set["pos"] = train_set["words"].apply(nltk.pos_tag)
    
    test_set = read_data(path=args.test_file)
    test_set = pd.DataFrame.from_dict(test_set)
    test_set["pos"] = test_set["words"].apply(nltk.pos_tag)

    # code options
    if args.option == 1:
        # pre-process the data to add additional features
        train = train_set
        
        # prepare the word lists and corresponding features
        word_to_ix = {}
        cap_to_ix  = {}
        plu_to_ix  = {}
        pos_to_ix  = {}
        tag_prev_to_ix  = {}

        texts   = train["words"]
        caps    = train["caps"]
        pos_tag = train["pos"]

        # check plural or not
        wnl = WordNetLemmatizer()
        def isplural(word):
            lemma = wnl.lemmatize(word, 'n')
            plural = 1 if word is not lemma else 0
            return plural

        # word list
        for sentence in texts:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        word_to_ix["None"] = len(word_to_ix)

        # capitalized list
        for cap in caps:
            for letter in cap:
                if letter not in cap_to_ix:
                    cap_to_ix[letter] = len(cap_to_ix) + len(word_to_ix)

        # check previous list
        for pos in pos_tag:
            for p in pos:
                if p[1] not in pos_to_ix:
                    pos_to_ix[p[1]] = len(pos_to_ix) + len(cap_to_ix) + len(word_to_ix)

        pos_to_ix["Start"] = len(pos_to_ix) + len(cap_to_ix) + len(word_to_ix)
        pos_to_ix["None"] = len(pos_to_ix) + len(cap_to_ix) + len(word_to_ix)

        # prepare the tag index
        START_TAG = "<START>"
        STOP_TAG  = "<STOP>"
        tag_to_ix = {"O": 0, "T-NEG": 1, "T-POS": 2, "T-NEU": 3, START_TAG: 4}            

        # prepare the dictionary for the previous tag
        for num, item in enumerate(tag_to_ix):
            tag_prev_to_ix[item] = len(pos_to_ix) + len(cap_to_ix) + len(word_to_ix) + num
            
        
        # save the total length of embedding
        N = len(pos_to_ix) + len(cap_to_ix) + len(word_to_ix) + len(tag_prev_to_ix)
        
        ## build up the list of features for training
        training = list(zip(train.words, train.ts_raw_tags, train.caps, train.pos))
        context = []
        tag = []
        for sentence, tags, caps, pos in training:
            # word hidden init by sentence
            for index, word in enumerate(sentence):
                # char hidden init by word
                if index == 0:
                    tag_prev = tag_prev_to_ix['<START>']
                    pos_prev = pos_to_ix["Start"]
                else:
                    tag_prev = tag_prev_to_ix[tags[index-1]]
                    pos_prev = pos_to_ix[pos[index-1][1]]

                context.append([word_to_ix[word], cap_to_ix[caps[index]], pos_prev, isplural(word) + N, 
                                tag_prev])
                tag.append(tags[index])

        tag_to_ix = {"O": 0, "T-NEG": 1, "T-POS": 2, "T-NEU": 3}
        training = list(zip(context, tag))
        
        # model training process
        CONTEXT_SIZE = 5
        EMBEDDING_DIM = 6
        losses = []
        val_losses = []
        test_losses = []
        loss_function = nn.NLLLoss()
        model = NN(N+2, EMBEDDING_DIM, CONTEXT_SIZE, len(tag_to_ix))
        optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-3)

        for epoch in range(50):
            total_loss = torch.Tensor([0])
            for sentence, tag in training:
                context_idxs = sentence
                context_var = autograd.Variable(torch.LongTensor(context_idxs))
                
                model.zero_grad()
                log_probs = model(context_var)

                loss = loss_function(log_probs, autograd.Variable(
                    torch.LongTensor([tag_to_ix[tag]])))
                loss.backward()
                optimizer.step()

                total_loss += loss.data
            losses.append(total_loss)

            print("Epoch No.", epoch)
        
        # prediction/decode section
        testing = list(zip(test_set.words, test_set.ts_raw_tags, test_set.caps, test_set.pos))
        sentence_encode = []
        tag_encode = []
        for sentence, tags, caps, pos in testing:
            context_test1 = []
            tag_test1 = []
            # word hidden init by sentence
            for index, word in enumerate(sentence):
                # char hidden init by word
                if word in word_to_ix:
                    word_lst = word_to_ix[word]
                else:
                    word_lst = word_to_ix["None"]

                if word in word_to_ix:
                    word_lst = word_to_ix[word]
                else:
                    word_lst = word_to_ix["None"]

                if index == 0:
                    tag_prev = tag_prev_to_ix['<START>']
                    pos_prev = pos_to_ix["Start"]
                else:
                    tag_prev = tag_prev_to_ix[tags[index-1]]

                    if pos[index-1] in pos_to_ix:
                        pos_prev = pos_to_ix[pos[index-1][1]]
                    else:
                        pos_prev = pos_to_ix["None"]

                context_test1.append([word_lst, cap_to_ix[caps[index]], pos_prev, isplural(word) + N, tag_prev])
                tag_test1.append(tags[index])
            sentence_encode.append(context_test1)
            tag_encode.append(tag_test1)
        
        # run Viterbi to decode and get scores
        decode_to_ix = {0: "O", 1: 'T-NEG', 2: 'T-POS', 3:'T-NEU'}
        def viterbi(sequence, decode_to_ix):
            decode = []
            for item in sequence:
                item = torch.LongTensor(item)
                prediction = model(item)
                target = torch.argmax(prediction, dim=1)
                decode.append(decode_to_ix[target.item()])
            return decode

        predictions = []
        with torch.no_grad():
            for test_lst in sentence_encode:
                pred_target = viterbi(test_lst, decode_to_ix)
                predictions.append(pred_target)
        
        
        eval_lst = list(zip(predictions, tag_encode))
        
    elif args.option == 2:
        
        wv_from_bin = KeyedVectors.load_word2vec_format(datapath("/homes/cs577/hw2/w2v.bin"), binary=True)
        vector = wv_from_bin['man']
        #wv_from_bin = KeyedVectors.load_word2vec_format(datapath("C:/Users/test_/OneDrive - purdue.edu/NLP_2021S/HW2/w2v.bin"), binary=True)
        #vector = wv_from_bin['man']
        print("Successfully loading the word2vec")
        # prepare the data
        train = train_set
        
        word_to_ix = {}
        cap_to_ix  = {}
        plu_to_ix  = {}
        pos_to_ix  = {}
        tag_prev_to_ix  = {}


        texts   = train["words"]
        caps    = train["caps"]
        pos_tag = train["pos"]

        # check plural or not
        wnl = WordNetLemmatizer()
        def isplural(word):
            lemma = wnl.lemmatize(word, 'n')
            plural = 1 if word is not lemma else 0
            return plural

        # word list
        for sentence in texts:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        word_to_ix["None"] = len(word_to_ix)
        
        # capitalized list
        for cap in caps:
            for letter in cap:
                if letter not in cap_to_ix:
                    cap_to_ix[letter] = len(cap_to_ix)

        # check previous list
        for pos in pos_tag:
            for p in pos:
                if p[1] not in pos_to_ix:
                    pos_to_ix[p[1]] = len(pos_to_ix) + len(cap_to_ix)

        pos_to_ix["Start"] = len(pos_to_ix) + len(cap_to_ix)
        pos_to_ix["None"] = len(pos_to_ix) + len(cap_to_ix)

        # prepare the tag index
        START_TAG = "<START>"
        STOP_TAG  = "<STOP>"
        tag_to_ix = {"O": 0, "T-NEG": 1, "T-POS": 2, "T-NEU": 3, START_TAG: 4}            

        # prepare the dictionary for the previous tag
        for num, item in enumerate(tag_to_ix):
            tag_prev_to_ix[item] = len(pos_to_ix) + len(cap_to_ix) + num

        N = len(pos_to_ix) + len(cap_to_ix) + len(tag_prev_to_ix)
        
        ## build up the list of features for training
        training = list(zip(train.words, train.ts_raw_tags, train.caps, train.pos))
        context = []
        tag = []
        word_lst = []
        embed_size = vector.shape[0]
        
        for sentence, tags, caps, pos in training:
            # word hidden init by sentence
            for index, word in enumerate(sentence):
                # char hidden init by word
                try: 
                    word_eb = wv_from_bin[word]
                except KeyError:
                    word_eb = np.random.normal(scale=0.6, size=(embed_size, ))
                if index == 0:
                    tag_prev = tag_prev_to_ix['<START>']
                    pos_prev = pos_to_ix["Start"]
                else:
                    tag_prev = tag_prev_to_ix[tags[index-1]]
                    pos_prev = pos_to_ix[pos[index-1][1]]

                word_lst.append(word_eb)

                context.append([cap_to_ix[caps[index]], pos_prev, isplural(word) + N, 
                                tag_prev])
                tag.append(tags[index])
        training = list(zip(word_lst, context, tag))
        
        CONTEXT_SIZE = 4
        embedding_dim = 50
        vocab_size = N+2
        losses = []
        test_losses = []
        loss_function = nn.NLLLoss()

        NeuralNet = NeuralN(embedding_dim, vocab_size, CONTEXT_SIZE, len(tag_to_ix))
        optimizer = optim.SGD(NeuralNet.parameters(), lr=0.05, weight_decay=1e-4)

        for epoch in range(50):
            total_loss = torch.Tensor([0])
            for embed, feature ,tag in training:
                embed_var = autograd.Variable(torch.FloatTensor(embed))
                feature_idxs = feature
                feature_var = autograd.Variable(torch.LongTensor(feature_idxs))

                NeuralNet.zero_grad()

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = NeuralNet(embed_var, feature_var)

                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a variable)
                loss = loss_function(log_probs, autograd.Variable(
                    torch.LongTensor([tag_to_ix[tag]])))

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                total_loss += loss.data
            losses.append(total_loss)
            print("Epoch No.", epoch)
        
        # prediction/decode section
        testing = list(zip(test_set.words, test_set.ts_raw_tags, test_set.caps, test_set.pos))
        sentence_encode = []
        tag_encode = []
        word_encode = []
        for sentence, tags, caps, pos in testing:
            context_test1 = []
            tag_test1 = []
            word_test1 = []
            # word hidden init by sentence
            for index, word in enumerate(sentence):
                # char hidden init by word
                try: 
                    word_eb = wv_from_bin[word]
                except KeyError:
                    word_eb = np.random.normal(scale=0.6, size=(embed_size, ))

                if index == 0:
                    tag_prev = tag_prev_to_ix['<START>']
                    pos_prev = pos_to_ix["Start"]
                else:
                    tag_prev = tag_prev_to_ix[tags[index-1]]

                    if pos[index-1] in pos_to_ix:
                        pos_prev = pos_to_ix[pos[index-1][1]]
                    else:
                        pos_prev = pos_to_ix["None"]

                context_test1.append([cap_to_ix[caps[index]], pos_prev, isplural(word) + N, tag_prev])
                tag_test1.append(tags[index])
                word_test1.append(word_eb)
            sentence_encode.append(context_test1)
            tag_encode.append(tag_test1)
            word_encode.append(word_test1)
        
        # run Viterbi to decode and get scores
        viterbi_decode = list(zip(word_encode, sentence_encode))
        decode_to_ix = {0: "O", 1: 'T-NEG', 2: 'T-POS', 3:'T-NEU'}
        
        def viterbi(w2v, sequence, decode_to_ix):
            decode = []
            for i in range(len(sequence)):
                item = torch.FloatTensor(w2v[i])
                char_lst = torch.LongTensor(sequence[i])
                prediction = NeuralNet(item, char_lst)
                target = torch.argmax(prediction, dim=1)
                decode.append(decode_to_ix[target.item()])
            return decode

        predictions = []
        with torch.no_grad():
            for w2v, sequence in viterbi_decode:
                pred_target = viterbi(w2v, sequence, decode_to_ix)
                predictions.append(pred_target)
        eval_lst = list(zip(predictions, tag_encode))
        
    else:
        # prepare the word lists and corresponding features
        train = train_set
        
        word_to_ix = {}
        cap_to_ix  = {}
        plu_to_ix  = {}
        pos_to_ix  = {}
        tag_prev_to_ix  = {}

        texts   = train["words"]
        caps    = train["caps"]
        pos_tag = train["pos"]

        # check plural or not
        wnl = WordNetLemmatizer()
        def isplural(word):
            lemma = wnl.lemmatize(word, 'n')
            plural = 1 if word is not lemma else 0
            return plural

        # word list
        for sentence in texts:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        word_to_ix["None"] = len(word_to_ix)

        # capitalized list
        for cap in caps:
            for letter in cap:
                if letter not in cap_to_ix:
                    cap_to_ix[letter] = len(cap_to_ix)

        # check previous list
        for pos in pos_tag:
            for p in pos:
                if p[1] not in pos_to_ix:
                    pos_to_ix[p[1]] = len(pos_to_ix) + len(cap_to_ix)

        pos_to_ix["Start"] = len(pos_to_ix) + len(cap_to_ix)
        pos_to_ix["None"] = len(pos_to_ix) + len(cap_to_ix)

        # prepare the tag index
        START_TAG = "<START>"
        STOP_TAG  = "<STOP>"
        tag_to_ix = {"O": 0, "T-NEG": 1, "T-POS": 2, "T-NEU": 3, START_TAG: 4}            

        # prepare the dictionary for the previous tag
        for num, item in enumerate(tag_to_ix):
            tag_prev_to_ix[item] = len(pos_to_ix) + len(cap_to_ix) + num

        N = len(pos_to_ix) + len(cap_to_ix) + len(tag_prev_to_ix)
        
        ## build up the list of features for training
        training = list(zip(train.words, train.ts_raw_tags, train.caps, train.pos))
        sentence_lst = []
        feature_lst  = []
        tag_list     = []

        for sentence, tags, caps, pos in training:
            context = []
            tag = []
            word_lst = []
            # word hidden init by sentence
            for index, word in enumerate(sentence):
                # char hidden init by word
                try: 
                    word_eb = word_to_ix[word]
                except KeyError:
                    word_eb = word_to_ix["None"]
                if index == 0:
                    tag_prev = tag_prev_to_ix['<START>']
                    pos_prev = pos_to_ix["Start"]
                else:
                    tag_prev = tag_prev_to_ix[tags[index-1]]
                    pos_prev = pos_to_ix[pos[index-1][1]]

                word_lst.append(word_eb)

                context += [cap_to_ix[caps[index]], pos_prev, isplural(word) + N, 
                                tag_prev]
                tag.append(tags[index])

                #training = list(zip(word_lst, context, tag))
            sentence_lst.append(word_lst)
            feature_lst.append(context)
            tag_list.append(tag)
        
        training = list(zip(sentence_lst, feature_lst, tag_list))
        
        EMBEDDING_DIM = 12
        WORD_HIDDEN_DIM = 12

        FEAT_EMBEDDING_DIM = 6
        FEAT_HIDDEN_DIM    = 6


        model = BiLSTM(EMBEDDING_DIM, FEAT_EMBEDDING_DIM, WORD_HIDDEN_DIM, FEAT_HIDDEN_DIM, len(word_to_ix), N+2, len(tag_to_ix)-1)
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-3)

        with torch.no_grad():
            sent = torch.LongTensor(sentence_lst[0])
            feat = torch.LongTensor(feature_lst[0])
            tag_scores = model(sent, feat)
            print(tag_scores.shape)
            
        losses = []
        for epoch in range(50):  # again, normally you would NOT do 300 epochs, it is toy data
            total_loss = torch.Tensor([0])
            for sent, feat, tags in training:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                model.word_hidden = model.init_hidden(WORD_HIDDEN_DIM)
                model.feat_hidden = model.init_hidden(FEAT_HIDDEN_DIM)

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                sentence_in = torch.tensor(sent, dtype=torch.long)
                feature_in = torch.tensor(feat, dtype=torch.long)
                targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

                log_probs = model(sentence_in, feature_in)

                # Step 3. Run our forward pass.
                loss = loss_function(log_probs, targets)
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss += loss.data

            losses.append(total_loss)
            print(epoch)
            
        testing = list(zip(test_set.words, test_set.ts_raw_tags, test_set.caps, test_set.pos))
        sentence_test = []
        feature_test  = []
        tag_test      = []
    

        for sentence, tags, caps, pos in testing:
            context = []
            tag = []
            word_lst = []
            # word hidden init by sentence
            for index, word in enumerate(sentence):
                # char hidden init by word
                try: 
                    word_eb = word_to_ix[word]
                except KeyError:
                    word_eb = word_to_ix["None"]
                if index == 0:
                    tag_prev = tag_prev_to_ix['<START>']
                    pos_prev = pos_to_ix["Start"]
                else:
                    tag_prev = tag_prev_to_ix[tags[index-1]]
                    pos_prev = pos_to_ix[pos[index-1][1]]

                word_lst.append(word_eb)

                context += [cap_to_ix[caps[index]], pos_prev, isplural(word) + N, 
                                tag_prev]
                tag.append(tags[index])

                #training = list(zip(word_lst, context, tag))
            sentence_test.append(word_lst)
            feature_test.append(context)
            tag_test.append(tag)

        viterbi_decode = list(zip(sentence_test, feature_test))
        
        # run Viterbi to decode and get scores
        decode_to_ix = {0: "O", 1: 'T-NEG', 2: 'T-POS', 3:'T-NEU'}
        def viterbi(sequence_seq, feature_seq, decode_to_ix):
            decode = []
            sentence = torch.LongTensor(sequence_seq)
            feature  = torch.LongTensor(feature_seq)
            prediction = model(sentence, feature)
            target = torch.argmax(prediction, dim=1)
            decode = [decode_to_ix[i.item()] for i in target]
            return decode

        predictions = []
        with torch.no_grad():
            for test_lst, feature_lst in viterbi_decode:
                pred_target = viterbi(test_lst, feature_lst, decode_to_ix)
                predictions.append(pred_target)
        eval_lst = list(zip(predictions, tag_test))
    
    test_set_with_labels = pd.DataFrame(test_set.copy())
    # TODO: change this portion to save your predicted labels in the dataframe instead, here we are just saving the true tags. Make sure the format is the same!
    test_set_with_labels['ts_pred_tags'] = test_set_with_labels['ts_raw_tags']
    test_set_with_labels['ts_pred_tags'] = predictions
    # now, save the predictions in the file
    # you can change this function but make sure you do NOT change the format of the file that is saved
    SaveFile(test_set_with_labels, output_csv_file=args.test_predictions_file)
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(eval_lst)):
        pred = eval_lst[i][0]
        orig = eval_lst[i][1]
        for j in range(len(pred)):
            if orig[j] == "O":
                if pred[j] != "O":
                    FP += 1
            else:
                if orig[j] == pred[j]:
                    TP += 1
                elif pred[j] == "O":
                    FN += 1
                else:
                    FP += 1
                    FN += 1

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = precision*recall*2/(precision+recall)
        

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", F1)
        
        
        # now, you must parse the dataset


# In[32]:


if __name__ == '__main__':
    main()


# In[ ]:




