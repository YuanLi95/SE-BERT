# -*- coding: utf-8 -*-

import numpy as np
import pickle
import spacy
import networkx as nx
import  re

from pytorch_pretrained_bert import  BertTokenizer
bert_tokenizer =BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

depenpency_type = "spacy"

if depenpency_type =="spacy":
    nlp = spacy.load('en_core_web_trf')


if depenpency_type=="stanza":
    import spacy_stanza
    nlp = spacy_stanza.load_pipeline("en")
if depenpency_type=="benepar":
    import benepar
    nlp = spacy.load('en_core_web_trf')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

token_nize_for_tokenize = spacy.load('en_core_web_trf')
def tokenize(text):
    text=text.strip()
    text=re.sub(r' {2,}',' ',text)
    text= re.sub(' ´ ','\'' ,text)
    text = text.strip()

    document = token_nize_for_tokenize(text)
    return " ".join([token.text for token in document])

def aspect_short_path(G, target):
    """"
    """
    d = nx.shortest_path_length(G, target=target)
    distance_list = []
    for node in G.nodes():
        try:
            distance_list.append(d[node])
        except KeyError:
            distance_list.append(-1)
    return distance_list

def dependency_adj_matrix(text,aspect_double_idx):
    # text = "Great food but the service was dreadful !"

    word_pieces = bert_tokenizer.tokenize(text)
    # for i in range
    # print("+++++++++++++++++++++++++++begin+++++++++++++++++++")
    try:
        document = nlp(text)

    except:
        document = token_nize_for_tokenize(text)
    #     document = token_nize_for_tokenize (text)
    seq_len = len(word_pieces)
    Syntactic_dependence = []
    # 创建三元组(piece_token,old_index,new_index)
    three_list = []
    old_index = 0

    for new_index in range(len(word_pieces)):
        if word_pieces[new_index][0:2] == "##":
            old_index = old_index - 1
        three_list.append([word_pieces[new_index], old_index, new_index])
        old_index = old_index + 1

    matrix_dir = np.zeros([seq_len,seq_len]).astype('float32')

    matrix_undir = np.zeros([seq_len, seq_len]).astype('float32')
    matrix_redir = np.zeros([seq_len, seq_len]).astype('float32')
    #加上CLS和SEP
    final_matrix_dir = np.ones([seq_len+2,seq_len+2]).astype('float32')
    # final_matrix_dir = np.zeros([seq_len+2,seq_len+2]).astype('float32')


    final_matrix_undir = np.ones([seq_len+2, seq_len+2]).astype('float32')
    # final_matrix_undir = np.zeros([seq_len + 2, seq_len + 2]).astype('float32')
    #
    final_matrix_redir = np.ones([seq_len+2, seq_len+2]).astype('float32')
    # final_matrix_redir = np.zeros([seq_len+2, seq_len+2]).astype('float32')




    for token in document:
        token_word_piece_list = []
        for _, old_index, new_index in three_list:
            if token.i == old_index:
                token_word_piece_list.append(new_index)
            # child 和 token 本身为word_piece, 分别找到新的 list
        head_word_piece_list = []
        for _, old_index, new_index in three_list:
            if token.head.i == old_index:
                head_word_piece_list.append(new_index)
        for token_piece in token_word_piece_list:
            for head_piece in head_word_piece_list:
                if head_piece ==token_piece:
                    continue
                matrix_undir[token_piece][head_piece] = 1
                matrix_undir[head_piece][token_piece] = 1
                matrix_dir[token_piece][head_piece] = 1
                matrix_redir[head_piece][token_piece] = 1
                #加1为了CLS

                Syntactic_dependence.append([token_piece+1, token.dep_.lower()+token.pos_.lower(), head_piece+1])
                # Syntactic_dependence.append([token_piece+1, token.dep_.lower(), head_piece+1])

                # Syntactic_dependence.append([head_piece+1, token.dep_.lower(), token_piece+1])
            Syntactic_dependence.append([token_piece + 1, "selfcycle", token_piece + 1])
    for i in range(len(matrix_undir)):
        matrix_undir[i][i]=1
        matrix_dir[i][i]=1
        matrix_redir[i][i]=1

    #中间
    final_matrix_dir[1:-1,1:-1] =matrix_dir
    final_matrix_undir[1:-1,1:-1] = matrix_undir
    final_matrix_redir[1:-1,1:-1] = matrix_redir


    G = nx.from_numpy_matrix(matrix_undir)
    aspect_begin_idx = aspect_double_idx[0]
    aspect_end_idx = aspect_double_idx[1]
    distance_aspect_begin = np.array(aspect_short_path(G, aspect_begin_idx))
    distance_aspect_end = np.array(aspect_short_path(G, aspect_end_idx))
    distance_aspect = np.array((distance_aspect_begin + distance_aspect_end)/2).astype(np.int32)
    distance_aspect[aspect_double_idx] = 0
    distance_aspect[aspect_end_idx]=0
    for index,distance in enumerate(distance_aspect):
        if distance ==-1:
            distance_aspect[index] = 1000
    return final_matrix_dir,final_matrix_redir,final_matrix_undir,Syntactic_dependence,distance_aspect


def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph_dir = {}
    idx2graph_redir ={}
    idx2graph_undir ={}
    Syntactic_dependence_all = {}
    idx2positon = {}
    # fout_dir= open(filename+"_"+depenpency_type +'dir'+ '.graph', 'wb')
    # fout_redir = open(filename+"_"+depenpency_type+"redir"+ '.graph', 'wb')
    fout_undir = open(filename+"_"+depenpency_type+'undir'+ '.graph', 'wb')
    dependency_analysis = open(filename+"_"+depenpency_type+'.dependency','wb')
    fout_syntax_position = open(filename+"_"+depenpency_type+'.syntax', 'wb')

    for i in range(0, len(lines), 3):
        print(i)

        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        text_left = tokenize(text_left)
        text_right = tokenize(text_right)


        aspect = lines[i + 1].lower().strip()
        aspect = tokenize(aspect)
        aspect_list = bert_tokenizer.tokenize(aspect)
        text_left_list = bert_tokenizer.tokenize(text_left)

        aspect_double_idx = [len(text_left_list), len(text_left_list)+len(aspect_list)-1]
        # print(bert_tokenizer.tokenize(text_left + ' ' + aspect + ' ' + text_right))
        # print(aspect)
        # print(aspect_double_idx)


        input_text = tokenize(text_left + ' ' + aspect + ' ' + text_right)
        # print(input_text)
        adj_matrix_dir,adj_matrix_redir, adj_matrix_undir,Syntactic_dependence,distance_aspect = dependency_adj_matrix(input_text,aspect_double_idx)
        idx2graph_dir[i] = adj_matrix_dir
        idx2graph_redir[i] = adj_matrix_redir
        idx2graph_undir[i] = adj_matrix_undir

        Syntactic_dependence_all[i] = Syntactic_dependence
        #syntax_position_distance
        idx2positon[i] = distance_aspect

    # pickle.dump(idx2graph_dir, fout_dir)
    pickle.dump(idx2graph_undir, fout_undir)
    # pickle.dump(idx2graph_redir,fout_redir)
    pickle.dump(Syntactic_dependence_all,dependency_analysis)
    pickle.dump(idx2positon,fout_syntax_position)
    # fout_dir.close()
    # fout_redir.close()
    fout_undir.close()
    dependency_analysis.close()
    fout_syntax_position.close()

if __name__ == '__main__':
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    print("------------------semeval_rest14--------------------")
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    print("------------------semeval_rest14--------------------")
    process('./datasets/semeval14/laptop_train.raw')

    process('./datasets/semeval14/laptop_test.raw')
    print("------------------semeval14_lap14--------------------")
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    print("------------------semeval15--------------------")
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')
    print("------------------semeval16--------------------")
    # # fin= open("./datasets/acl-14-short-data/train.raw.dependency","rb")

