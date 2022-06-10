# -*- coding: utf-8 -*-
import argparse
import random
import numpy
import torch

import codecs
from Trainer import  Instructor
from models.dependency_bert import  BertForSequenceClassification as Dependecy_Bert
from models.dependecy_bert_mutilayer import BertForSequenceClassification as De_Bert_Muti
from  models.bilinear_DB_bert import BertForSequenceClassification as Bilinear_DB
from  models.ensemble_DB_Bert import BertForSequenceClassification as EnsembleDB
from  models.bert_base import BertForSequenceClassification as Bert_base

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='EnsembleDB', type=str)  # De_Bert_Muti #bilinear,Bilinear_DB
    parser.add_argument('--dataset', default='rest15', type=str, help='lap14,twitter, rest14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.000005, type=float)
    parser.add_argument('--l2reg', default=0.0001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--position_dim',default=768,type=int)
    parser.add_argument('--position_drop', default=0.4, type=float)
    parser.add_argument('--dependency_edge_dim',default=100,type=int)

    parser.add_argument('--pretrained_weights', default="bert-base-uncased", type=str)
    parser.add_argument('--num_labels', default=3, type=int)
    parser.add_argument('--K_alpha', default=0, type=float)
    parser.add_argument('--V_alpha', default=0, type=float)
    parser.add_argument('--n_gpu', default=1, type=int)



    # parser.add_argument('--gat_hidden_dim',default=100,type=int )
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--seed', default=9, type=int)  # 776
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--use_lstm_attention', default=True, type=bool)
    parser.add_argument('--use_bert', default=False, type=bool)
    parser.add_argument('--use_speech_weight', default=True, type=bool)
    parser.add_argument('--lcf', default="cdw",type=str)
    parser.add_argument('--num_hidden_layers',default=12,type=int)
    parser.add_argument('--SRD',default=3,type=int)
    parser.add_argument('--lr_de', default=0.5, type=float)
    parser.add_argument('--DB_begin_layer', default=12, type=int)
    parser.add_argument('--edge_type_number',default=-1,type=int)
    parser.add_argument('--count', default=0, type=int)
    parser.add_argument('--dependency_type', default="spacy_only", type=str)
    parser.add_argument('--redroop_alpha', default=1.0, type=float)


    # Add_DT_layer
    # parser.add_argument('--DT_layer_number', default=6, type=int)

    opt = parser.parse_args()
    model_classes = {

        'De_Bert': Dependecy_Bert,
        'De_Bert_Muti':De_Bert_Muti,
        "Bilinear_DB" :Bilinear_DB,
        "EnsembleDB":EnsembleDB,
        "Bert_base": Bert_base,
    }
    input_colses = {
        'De_Bert': ["text_indices","aspect_indices","left_indices","dependency_graph_undir","dependency_edge_undir","position_syntax_indices","aspect_double_idx","attention_mask"],
        'De_Bert_Muti': ["text_indices", "aspect_indices", "left_indices", "dependency_graph_undir",
                    "dependency_edge_undir", "position_syntax_indices", "aspect_double_idx", "attention_mask"],
        'Bilinear_DB': ["text_indices", "aspect_indices", "left_indices", "dependency_graph_undir",
                         "dependency_edge_undir", "position_syntax_indices", "aspect_double_idx",
                         "attention_mask"],
        'EnsembleDB': ["text_indices", "aspect_indices", "left_indices", "dependency_graph_undir",
                        "dependency_edge_undir", "position_syntax_indices", "aspect_double_idx",
                        "attention_mask"],
        'Bert_base': ["text_indices", "aspect_indices", "left_indices", "dependency_graph_undir",
                       "dependency_edge_undir", "position_syntax_indices", "aspect_double_idx",
                       "attention_mask"],
    }

    input_dependency_type ={
        "spacy_only":["spacy"],
        "stanza_only":["stanza"],
        "benepar_only":["benepar"],
        "stanza_spacy": ["stanza","spacy"],
        "stanza_benepar": ["stanza", "benepar"],
        "spacy_benepar": ["spacy", "benepar"],
        "ensemble":["spacy","stanza","benepar"]
    }


    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }


    opt.model_class = model_classes[opt.model_name]
    # summary(opt.model_class,input_size=(32,32,300))
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]


    opt.syntactic_parsing = input_dependency_type[opt.dependency_type]
    opt.edge_type_number = len(opt.syntactic_parsing)

    # opt.device = torch.device('cpu')
    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # opt.device = torch.device('cpu')
    # print(opt.device)



    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
    repeats = 3
    max_test_acc_avg = 0
    max_test_f1_avg = 0

    for i in range(repeats):
        opt.repeat=i
        ins = Instructor(opt)
        max_test_acc,max_test_f1=ins.run()
        max_test_acc_avg += max_test_acc
        max_test_f1_avg += max_test_f1
    print("this is repeat {0}".format(i))
    f_out = codecs.open('log/' + opt.model_name + '_' + opt.dataset  +'dependency_type'+str(opt.dependency_type)+ '_val.txt', 'a+',encoding="utf-8")
    print("max_test_acc_avg:", max_test_acc_avg / repeats)
    print("max_test_f1_avg:", max_test_f1_avg / repeats)
    f_out.write('max_test_acc_avg: {0}, max_test_f1_avg: {1}\n'.format(max_test_acc_avg / repeats,
                                                                       max_test_f1_avg / repeats))
    f_out.write("\n")


