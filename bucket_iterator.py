# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy
def get_attn_pad_mask(seq_q, seq_k, pad_index=0):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = (seq_k.data!=pad_index).unsqueeze(1)
    pad_attn_mask = torch.as_tensor(pad_attn_mask, dtype=torch.int)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def replace_aspect_operation( batch_mask,dependency_graph, aspect_double_idx):
    """
    #将aspect的mask代替为依赖树结构
    """
    batch_size, seq_len = batch_mask.shape[0], batch_mask.shape[1]
    replaced_mask = batch_mask.numpy().tolist()
    dependency_graph = dependency_graph.numpy().astype(int).tolist()
    for i in range(batch_size):
        begin = (aspect_double_idx[i][0].int()).data
        end = (aspect_double_idx[i][1].int()+1).data
        for index in range(begin,end):
            replaced_mask[i][index]=dependency_graph[i][index]
    replaced_mask = numpy.array(replaced_mask)
    return replaced_mask



class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True,syntactic_tool=None):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.syntactic_tool = syntactic_tool
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph_undir_all = {}
        batch_dependency_edge_undir_all = {}
        batch_mask_all = {}
        for syntactic_name in self.syntactic_tool:
            batch_dependency_graph_undir_all[syntactic_name ]=[]
            batch_dependency_edge_undir_all[syntactic_name ] = []
        batch_position_syntax_indices =[]
        batch_aspect_double_idx = []
        # for syntactic_name in self.syntactic_tool:




        max_len = max([len(t[self.sort_key]) for t in batch_data])
        # print(max_len)
        for item in batch_data:

            text_indices, context_indices, aspect_indices, left_indices, polarity,dependency_graph_undir_all,dependency_edge_undir_all,\
            position_syntax_matrix= item['text_indices'], item['context_indices'], item['aspect_indices'], item['left_indices'],\
                item['polarity'], item['dependency_graph_undir_all'],item['dependency_edge_matrix_undir_all'],item['position_syntax_matrix'],

            text_padding = [0] * (max_len - len(text_indices))
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))

            text_indices_finaly = text_indices + text_padding



            position_syntax_matrix =position_syntax_matrix + text_padding
            context_indices = context_indices + context_padding
            aspect_indices = aspect_indices + aspect_padding
            left_indices = left_indices + left_padding

            batch_text_indices.append(text_indices_finaly)
            batch_position_syntax_indices.append(position_syntax_matrix)
            batch_context_indices.append(context_indices )
            batch_aspect_indices.append(aspect_indices )
            batch_left_indices.append(left_indices)
            batch_polarity.append(polarity)

            #将不同的依赖句法给分配到不同的list
            for syntactic_name in self.syntactic_tool:
                dependency_graph_undir = numpy.pad(dependency_graph_undir_all[syntactic_name],\
                     ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),'constant')
                batch_dependency_graph_undir_all[syntactic_name].append(dependency_graph_undir )

                dependency_edge_undir = numpy.pad(dependency_edge_undir_all[syntactic_name], \
                                                   ((0, max_len - len(text_indices)), (0, max_len - len(text_indices))),
                                                   'constant')
                batch_dependency_edge_undir_all[syntactic_name].append(dependency_edge_undir)


        batch_text_indices = torch.tensor(batch_text_indices)

        batch_context_indices = torch.tensor(batch_context_indices)
        batch_aspect_indices = torch.tensor(batch_aspect_indices)
        batch_left_indices = torch.tensor(batch_left_indices)
        batch_polarity = torch.tensor(batch_polarity)
        for syntactic_name in self.syntactic_tool:
            batch_dependency_graph_undir_all[syntactic_name] = torch.tensor(batch_dependency_graph_undir_all[syntactic_name])
            batch_dependency_edge_undir_all[syntactic_name] = torch.tensor(batch_dependency_edge_undir_all[syntactic_name]).long()

        batch_position_syntax_indices = torch.tensor(batch_position_syntax_indices)
        aspect_len = torch.sum(batch_aspect_indices != 0, dim=-1) - 2
        left_len = torch.sum(batch_left_indices != 0, dim=-1) - 1
        batch_aspect_double_idx = torch.cat([left_len.unsqueeze(1), \
                                       (left_len + aspect_len - 1).unsqueeze(1)], dim=1)

        for syntactic_name in self.syntactic_tool:

            batch_mask_all[syntactic_name] = get_attn_pad_mask(batch_text_indices,batch_text_indices)
            # print(batch_mask_all[syntactic_name])
            # print(batch_mask_all[syntactic_name])
            # print(batch_dependency_graph_undir_all[syntactic_name])
            # print( batch_aspect_double_idx)
            batch_mask_all[syntactic_name]  = torch.tensor(replace_aspect_operation(batch_mask_all[syntactic_name] ,batch_dependency_graph_undir_all[syntactic_name],batch_aspect_double_idx))
        # print(batch_dependency_graph_undir_all)
        # print(batch_dependency_graph_undir_all)
        # print(batch_dependency_edge_undir_all)
        # print(batch_dependency_edge_undir_all)
        # print(batch_mask_all)



        return { \
                'text_indices': batch_text_indices, \
                'context_indices': batch_context_indices, \
                'aspect_indices': batch_aspect_indices, \
                'left_indices': batch_left_indices, \
                'polarity': batch_polarity, \
                'dependency_graph_undir': batch_dependency_graph_undir_all, \
                'dependency_edge_undir': batch_dependency_edge_undir_all, \
                'position_syntax_indices':batch_position_syntax_indices,\
                'aspect_double_idx':batch_aspect_double_idx,\
                'attention_mask':batch_mask_all
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
