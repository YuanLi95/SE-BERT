# -*- coding: utf-8 -*-

import pickle
import numpy as np
import  os
import tqdm
import re
from transformers import BertTokenizer, BertModel
from denpendent_graph_new import  tokenize
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model= BertModel.from_pretrained('bert-base-uncased')

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
#    if os.path.exists(embedding_matrix_file_name):
#        print('loading embedding_matrix:', embedding_matrix_file_name)
#        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
#    else:
    if True:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


def build_dependency_matrix(dependency2idx, dependency_dim, type,dependency_type):
    embedding_matrix_file_name = '{0}_{1}_{2}_dependency_matrix.pkl'.format(str(type),dependency_dim,dependency_type)
    print(embedding_matrix_file_name)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading edge vectors ...')
        embedding_matrix = np.zeros((len(dependency2idx), dependency_dim))  # idx 0 and 1 are all-zeros
        # embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(dependency_dim), 1 / np.sqrt(dependency_dim), (1, dependency_dim))
        # embedding_matrix[1, :] = np.zeros(), (1, dependency_dim))

        print('building edge_matrix:', embedding_matrix_file_name)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix



def build_position_matrix(position2idx, position_dim, type,dependency_type):
    embedding_matrix_file_name = '{0}_{1}_{2}_position_matrix.pkl'.format(type, str(position_dim),dependency_type)

    embedding_matrix = np.zeros((len(position2idx), position_dim))  # idx 0 and 1 are all-zeros
    embedding_matrix[1, :] = np.random.uniform(-1 / np.sqrt(position_dim), 1 / np.sqrt(position_dim), (1, position_dim))
    # embedding_matrix[1, :] = np.random.uniform(-0.25, 0.25, (1, position_dim))


    print('building position_matrix:', embedding_matrix_file_name)
    pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Dependecynizer(object):
    def __init__(self, dependency2idx=None):
        if dependency2idx is None:
            self.dependency2idx = {}
            self.idx2dependency = {}
            self.idx2dependency_number={}
            self.idx = 0
            self.dependency2idx['<pad>'] = self.idx
            self.idx2dependency[self.idx] = '<pad>'
            self.idx2dependency_number['<pad>']=1
            self.idx += 1
            self.dependency2idx['<unk>'] = self.idx
            self.idx2dependency[self.idx] = '<unk>'
            self.idx2dependency_number['<unk>'] = 1
            self.idx += 1
            self.dependency2idx['sptype'] = self.idx
            self.idx2dependency[self.idx] = 'sptype'
            self.idx2dependency_number['sptype'] = 1
            self.idx += 1

        else:
            self.dependency2idx = dependency2idx
            self.idx2dependency = {v: k for k, v in dependency2idx.items()}
            self.idx2dependency_number = {v: k for k, v in dependency2idx.items()}
        self.idx2dependency_number = {}
    def fit_on_dependency(self, dependency_edge):
        dependency_edges = dependency_edge.lower()
        dependency_edges = dependency_edges.split()
        for dependency_edge in dependency_edges:
            if dependency_edge not in self.dependency2idx:
                self.dependency2idx[dependency_edge] = self.idx
                self.idx2dependency[self.idx] = dependency_edge
                self.idx2dependency_number[dependency_edge]=1
                self.idx += 1
            else:
                self.idx2dependency_number[dependency_edge] += 1
    def dependency_to_index(self,dependency_edge,idx2gragh_dir):
        edge_matrix = np.zeros_like(idx2gragh_dir,dtype=int)
        edge_matrix_re = np.zeros_like(idx2gragh_dir, dtype=int)
        edge_matrix_undir = np.zeros_like(idx2gragh_dir, dtype= int)
        matrix_len = (edge_matrix.shape)[0]

        unknownidx = 1
        for i in dependency_edge:
            try:
                if (matrix_len>int(i[0]))&(matrix_len>int(i[2])):
                    edge_matrix[i[0]][i[2]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx
                    edge_matrix_re[i[2]][i[0]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx

                    edge_matrix_undir[i[2]][i[0]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx
                    edge_matrix_undir[i[0]][i[2]] = self.dependency2idx[i[1]] if i[1] in self.dependency2idx else unknownidx

            except IndexError:
                print(matrix_len)
                print(dependency_edge)
        edge_matrix_undir[0,:] = [0]*edge_matrix_undir.shape[0]
        edge_matrix_undir[:,-1] = [0] * edge_matrix_undir.shape[0]
        edge_matrix_undir[:,0] = [0] * edge_matrix_undir.shape[0]
        edge_matrix_undir[-1,:] = [0] * edge_matrix_undir.shape[0]

        # return edge_matrix,edge_matrix_re,edge_matrix_undir

        return edge_matrix_undir


class Positionnizer(object):
    def __init__(self, position2idx=None):
        if position2idx is None:
            self.position2idx = {}
            self.idx2position = {}
            self.idx = 0
            self.position2idx['<pad>'] = self.idx
            self.idx2position[self.idx] = '<pad>'
            self.idx += 1
            self.position2idx['<unk>'] = self.idx
            self.idx2position[self.idx] = '<unk>'
            self.idx += 1
            self.position2idx['<CLS>'] = self.idx
            self.idx2position[self.idx] = '<CLS>'
            self.idx += 1
            self.position2idx['<SEP>'] = self.idx
            self.idx2position[self.idx] = '<SEP>'
            self.idx += 1
        else:
            self.position2idx = position2idx
            self.idx2position = {v: k for k, v in position2idx.items()}

    def fit_on_position(self, syntax_positions):
        for syntax_position in syntax_positions:
            if syntax_position not in self.position2idx:
                self.position2idx[syntax_position] = self.idx
                self.idx2position[self.idx] = syntax_position

                self.idx += 1
    def position_to_index(self,position_sequence):
        position_sequence = position_sequence.astype(np.str)
        unknownidx = 1
        position_matrix = [self.position2idx[w] if w in self.position2idx else unknownidx for w in position_sequence]
        # print(self.position2idx)
        position_matrix = [self.position2idx["<CLS>"]] + position_matrix+[self.position2idx["<SEP>"]]
        return position_matrix


class Tokenizer(object):
    def __init__(self, opt,word2idx=None, tokenizer=None):
        print(opt.pretrained_weights)
        self.tokenizer = BertTokenizer.from_pretrained(opt.pretrained_weights)
        self.pretrained_weights = opt.pretrained_weights
        print('load successfully')
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v: k for k, v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower().strip()
        words = tokenize(text)
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, tran=False):
        text = text.lower().strip()
        words = tokenize(text)
        words = words.split()
        trans = []
        realwords = []
        for word in words:
            wordpieces = self.tokenizer._tokenize(word)
            tmplen = len(realwords)

            realwords.extend(wordpieces)
            trans.append([tmplen, len(realwords)])
        #        unknownidx = 1_convert_token_to_id
        sequence = [self.tokenizer._convert_token_to_id('[CLS]')] + [self.tokenizer._convert_token_to_id(w) for w in
                                                                     realwords] + [
                       self.tokenizer._convert_token_to_id('[SEP]')]

        # sequence = [self.tokenizer._convert_token_to_id('[CLS]')] + [self.tokenizer._convert_token_to_id("[MASK]") for w in
        #                                                              realwords] + [
        #                self.tokenizer._convert_token_to_id('[SEP]')]
        if len(sequence) == 0:
            sequence = [0]
        if tran: return sequence, trans
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames,opt):
        text = ''
        dependency_all =''
        syntax_position_all= []
        print("________read_text__________")

        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            dependency_file_all = [[]]*(opt.edge_type_number)

            for index,syntactic_parsing in  enumerate(opt.syntactic_parsing):
                dependency_file = open(fname +"_"+syntactic_parsing+ ".dependency", 'rb')
                dependency_file_all[index] = pickle.load(dependency_file)
                dependency_file.close()

                # print(fname+".dependency")

            syntax_file = open(fname + ".syntax", "rb")
            syntax_position = pickle.load(syntax_file)
            syntax_file.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                # text += text_raw + " "
                text += text_raw + " "
                for dependency in dependency_file_all:
                    dependency_all+=  " ".join([i[1] for i in dependency[i]])
                    dependency_all+=" "
                for j in syntax_position[i]:
                    syntax_position_all.append(str(j))
            # print(syntax_position_all)
        return text,dependency_all,syntax_position_all

    @staticmethod
    def __read_data__(fname, tokenizer,dependency_tokenizer,position_tokenizer,opt):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()



        dependecy_file_all = {}
        idx2gragh_undir_all ={}
        for syntactic_name in opt.syntactic_parsing:
            fin_undir = open(fname +"_"+syntactic_name+'undir.graph', 'rb')
            idx2gragh_undir = pickle.load(fin_undir)
            idx2gragh_undir_all[syntactic_name] = idx2gragh_undir
            fin_undir.close()
            fin = open(fname +"_"+syntactic_name+'.dependency', 'rb')
            dependency_file = pickle.load(fin)
            dependecy_file_all[syntactic_name] = dependency_file
            fin.close()

        fin = open(fname + '.syntax', 'rb')
        position_syntax_file = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            print(i)
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            #tokenize 来自dependency_graph

            text = text_left + " " + aspect + " " + text_right
            context = text_left + " " + text_right
            aspect = aspect
            left = text_left
            text_indices = tokenizer.text_to_sequence(text)
            context_indices  = tokenizer.text_to_sequence(context)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(left)
            polarity = int(polarity) + 1
            #
            dependency_edge_matrix_undir_all = {}
            dependency_graph_undir_all={}


            for syntactic_name in opt.syntactic_parsing:
                dependency_graph_undir_i = idx2gragh_undir_all[syntactic_name][i]
                dependency_file_i = dependecy_file_all[syntactic_name][i]
                dependency_edge_matrix_undir_all[syntactic_name] = dependency_tokenizer.dependency_to_index(dependency_file_i,dependency_graph_undir_i)
                dependency_graph_undir_all[syntactic_name] = idx2gragh_undir_all[syntactic_name][i]

            position_syntax_matrix = position_tokenizer.position_to_index(position_syntax_file[i])


            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph_undir_all': dependency_graph_undir_all,
                'dependency_edge_matrix_undir_all': dependency_edge_matrix_undir_all,
                'position_syntax_matrix':position_syntax_matrix,
            }

            all_data.append(data)
        return all_data

    def __init__(self, position_dim,dataset='rest14', embed_dim=300,dependency_dim=100,use_bert=False,max_len=70,opt=None):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },

        }
        text,dependency_all,syntax_position_all = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']],opt)
        # print(dependency_all)
        # print(text)
        # exit()
        self.tokenizer = Tokenizer(opt)
        # if os.path.exists(dataset + '_word2idx.pkl'):
        #     print("loading {0} tokenizer...".format(dataset))
        #     with open(dataset + '_word2idx.pkl', 'rb') as f:
        #         word2idx = pickle.load(f)
        #         tokenizer = Tokenizer(word2idx=word2idx)
        # else:
        #     tokenizer = Tokenizer()
        #     tokenizer.fit_on_text(text)
        #     with open(dataset + '_word2idx.pkl', 'wb') as f:
        #         pickle.dump(tokenizer.word2idx, f)
         #构建边的映射关系

        pkl_dataset = "./pkl/"+dataset
        if os.path.exists(pkl_dataset +"_"+opt.dependency_type+'_dependency2idx.pkl'):
            print("loading {0} tokenizer...".format(pkl_dataset))
            with open(pkl_dataset + "_"+opt.dependency_type+'_dependency2idx.pkl', 'rb') as f:
                dependency2idx = pickle.load(f)
                dependency_tokenizer = Dependecynizer(dependency2idx=dependency2idx)
                #
        else:

            dependency_tokenizer = Dependecynizer()
            dependency_tokenizer.fit_on_dependency (dependency_all)
            with open(pkl_dataset +"_"+opt.dependency_type+ '_dependency2idx.pkl', 'wb') as f:
                pickle.dump(dependency_tokenizer.dependency2idx, f)
                print(dependency_tokenizer.dependency2idx)
        # print(dependency_tokenizer.dependency2idx)
        # exit()
        # 构建position的映射关系
        if os.path.exists(pkl_dataset + '_position2idx.pkl'):
            print("loading {0} position_tokenizer...".format(pkl_dataset))
            with open(pkl_dataset + '_position2idx.pkl', 'rb') as f:
                position2idx = pickle.load(f)
                position_tokenizer = Positionnizer(position2idx=position2idx )
        else:
            position_tokenizer = Positionnizer()
            position_tokenizer.fit_on_position(syntax_position_all)
            with open(pkl_dataset + '_position2idx.pkl', 'wb') as f:
                pickle.dump(position_tokenizer.position2idx, f)
        # self.denpendecy_tokenizer = dependency_tokenizer




        print(dependency_tokenizer.dependency2idx)


        self.dependency_matrix = build_dependency_matrix(dependency_tokenizer.dependency2idx,dependency_dim,pkl_dataset,opt.dependency_type)

        self.position_matrix = build_position_matrix(position_tokenizer.position2idx, position_dim, pkl_dataset,opt.dependency_type)


        print("-------------------------------------------build_all_matrix---------------------------------------")

        if os.path.exists(pkl_dataset + "_"+opt.dependency_type+'_train_data.pkl'):
            print("loading {0} position_tokenizer...".format(pkl_dataset))
            with open(pkl_dataset + "_"+opt.dependency_type+'_train_data.pkl', 'rb') as f:
                self.train_data = pickle.load(f)
        else:
            self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], self.tokenizer,dependency_tokenizer,position_tokenizer,opt))
            with open(pkl_dataset +"_"+opt.dependency_type+'_train_data.pkl', 'wb') as f:
                pickle.dump(self.train_data, f)

        if os.path.exists(pkl_dataset + "_"+opt.dependency_type+'_test_data.pkl'):
            print("loading {0} position_tokenizer...".format(pkl_dataset))
            with open(pkl_dataset + "_"+opt.dependency_type+'_test_data.pkl', 'rb') as f:
                self.test_data = pickle.load(f)
        else:
            self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], self.tokenizer,dependency_tokenizer,position_tokenizer,opt))
            with open(pkl_dataset +"_"+opt.dependency_type+'_test_data.pkl', 'wb') as f:
                pickle.dump(self.test_data, f)