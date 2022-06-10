import math

import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
import  time
import codecs
from models.get_optim import get_Adam_optim, get_Adam_optim_v2
import  datetime
import  torch.nn.functional as F
import pickle

def compute_kl_loss( prediction,label, pad_mask = None):

    p_loss = F.kl_div(F.log_softmax(prediction, dim=-1), F.softmax(label, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(label, dim=-1), F.softmax(prediction, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return  loss

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        pretrained_weights = opt.pretrained_weights
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim,dependency_dim=opt.dependency_edge_dim,position_dim=opt.position_dim, use_bert=opt.use_bert, max_len=70,opt=opt)
        self.absa_dataset= absa_dataset
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True,sort=True,syntactic_tool = opt.syntactic_parsing,
                                               )
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=1, shuffle=False,sort=False,syntactic_tool= opt.syntactic_parsing,
                                               )
        opt.dependency_matrix = absa_dataset.dependency_matrix
        opt.position_matrix = absa_dataset.position_matrix

        self.model = opt.model_class.from_pretrained(pretrained_weights, num_labels=opt.num_labels, cus_config=opt)
        # self.model =None
        # model_state = "./ada_pretrain_bert/epoch_2_saved.bin"
        # state_dict = torch.load(model_state,map_location="cuda:0")
        # self.model.load_state_dict(state_dict)
        if self.opt.n_gpu > 1:
            self.model  = torch.nn.DataParallel(self.model).to(self.opt.device)
            self.optim = get_Adam_optim_v2(opt, self.model.module)
        else:
            self.model = self.model.to(self.opt.device)
            self.optim = get_Adam_optim_v2(opt, self.model)

        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            if ((arg=="position_matrix")==True)|((arg=="dependency_matrix")==True):
                continue
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
       for p in self.model.parameters():
           if p.requires_grad:
               if len(p.shape) > 1:
                   self.opt.initializer(p)
               else:
                   stdv = 1. / math.sqrt(p.shape[0])
                   torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def label_smoothing(self, inputs, epsilon=0.1):
        V = inputs.get_shape().as_list()[-1]  # number of channels
        return ((1 - epsilon) * inputs) + (epsilon / V)




    def _evaluate_acc_f1(self,criterion):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total,test_loss_all = 0, 0,0
        t_targets_all, t_outputs_all, = None, None,
        attention_all = []
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_need_input=[]
                for col in self.opt.inputs_cols:
                    if type(t_sample_batched[col]) != dict:
                        t_need_input.append(t_sample_batched[col].to(self.opt.device))
                    else:
                        # 将多个dependency_tool 放入
                        for key, value in t_sample_batched[col].items():
                            t_sample_batched[col][key] = value.to(self.opt.device)
                        t_need_input.append(t_sample_batched[col])
                text_indices, aspect_indices, left_indices, dependency_graph_undir, \
                dependency_edge_undir, position_syntax_indices, \
                aspect_double_idx, _ = t_need_input
                attention_mask = (text_indices != 0).long().to(self.opt.device)

                t_targets = t_sample_batched['polarity'].to(self.opt.device)

                outputs_all = self.model(
                    input_ids=text_indices,
                    graph_undir=dependency_graph_undir,
                    edge_matrix_undir=dependency_edge_undir,
                    position_syntax_indices=position_syntax_indices,
                    aspect_double_idx=aspect_double_idx,
                    attention_mask=attention_mask,
                )
                t_outputs = outputs_all[1]

                # print(t_outputs)
                # print(t_targets)
                # print("--------------------------------------")
                # print(torch.argmax(t_outputs, -1).cpu())
                test_loss = criterion(t_outputs,t_targets)
                test_loss_all += test_loss.item()
                # print(t_outputs)
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        # print(len(attention_all))
        # print(t_targets_all.shape)
        test_acc = n_test_correct / n_test_total
        test_loss_all /= n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        return test_acc, f1,test_loss_all,t_targets_all.cpu(),torch.argmax(t_outputs_all, -1).cpu(),attention_all

    def _train(self, criterion):
        opt = self.opt
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        continue_not_increase = 0

        for epoch in range(self.opt.num_epoch):
            increase_flag = False
            print('>' * 100)
            print('epoch: ', epoch)

            n_correct, n_total = 0, 0

            n_train_correct, n_train_total, trian_loss_all = 0, 0, 0
            time_start = time.time()
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                self.model.train()
                self.optim.zero_grad()


                need_input = []
                for col in self.opt.inputs_cols:
                    if type(sample_batched[col])!= dict:
                        need_input.append(sample_batched[col].to(self.opt.device))
                    else:
                        # 将多个dependency_tool 放入
                        for key,value in sample_batched[col].items():
                            sample_batched[col][key]=value.to(self.opt.device)
                        need_input.append(sample_batched[col])
                text_indices, aspect_indices, left_indices, dependency_graph_undir, \
                dependency_edge_undir, position_syntax_indices, \
                aspect_double_idx,_ = need_input
                attention_mask = (text_indices != 0).long().to(self.opt.device)  # id of [PAD] is 0


                # attention_mask = dependency_graph_undir   # 用 graph 来代替attention mask


                targets = sample_batched['polarity'].to(self.opt.device)

                outputs_all_1= self.model(
                    input_ids=text_indices,
                    graph_undir=dependency_graph_undir,
                    edge_matrix_undir=dependency_edge_undir,
                    position_syntax_indices=position_syntax_indices,
                    aspect_double_idx=aspect_double_idx,
                    attention_mask=attention_mask,
            )
                # outputs_all_2 = self.model(
                #     input_ids=text_indices,
                #     graph_undir=dependency_graph_undir,
                #     edge_matrix_undir=dependency_edge_undir,
                #     position_syntax_indices=position_syntax_indices,
                #     aspect_double_idx=aspect_double_idx,
                #     attention_mask=attention_mask,
                # )
                # outputs = outputs_all_1[1]
                outputs_1 = outputs_all_1[1]
                # outputs_2 = outputs_all_2[1]
                ce_loss = criterion(outputs_1, targets)

                # kl_loss = compute_kl_loss(outputs_1, outputs_2)
                # loss = ce_loss +  opt.redroop_alpha* kl_loss
                loss = ce_loss
                # exit()

                # print(outputs)
                # print(targets)
                # print(torch.argmax(outputs, -1).cpu())

                # print(targets.shape)

                # loss = criterion(outputs, targets)
                trian_loss_all += loss.item()

                loss.backward()
                self.optim.step()
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs_1, -1) == targets).sum().item()
                    n_total += len(outputs_1)
                    train_acc = n_correct / n_total

                    test_acc, test_f1, test_loss,t_targets_all, t_outputs_all,alpha_attention = self._evaluate_acc_f1(criterion)
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1

                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            torch.save(self.model,
                                       './state_dict/test/' +"_"+ self.opt.model_name+self.opt.dependency_type + '_' + self.opt.dataset+'_'+"count:{0}_repeat:{1}".format(self.opt.count,self.opt.repeat)+'.pkl')
                            print('>>> this best model saved.this f1 is {:.4f}'.format(max_test_f1))
                            report = metrics.classification_report(t_targets_all.cpu(),
                                                                   t_outputs_all.cpu(),
                                                                   labels=[0, 1, 2], )
                            print(report)

                    print('\r >>> this repeat f1 is {:.4f}'.format(max_test_f1 ))
                    print('\r lr:{:E} train_loss_all: {:.4f}, acc: {:.4f}, test_loss_all{:.4f}，test_acc: {:.4f}, test_f1: {:.4f}'.format(self.optim.param_groups[0]['lr'],trian_loss_all/global_step, train_acc,
                                                                                                           test_loss,test_acc, test_f1))

            time_end = time.time()
            time_consule = time_end - time_start
            print("----------this epoch time :{0}".format(str(datetime.timedelta(seconds=time_consule))))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase%3==0:
                    new_lr = opt.lr_de*self.optim.param_groups[0]['lr']
                    self.optim.param_groups[0]['lr'] = new_lr
                if continue_not_increase >= 5:
                    print('early stop.')
                    return max_test_acc, max_test_f1
                    break
            else:
                continue_not_increase = 0

        return max_test_acc, max_test_f1


    def get_evaluate_attention(self,path):
        # switch model to evaluation mode
        self.model = torch.load(path)
        self.model.eval()
        n_test_correct, n_test_total,test_loss_all = 0, 0,0
        t_targets_all, t_outputs_all, = None, None,
        attention_all = []

        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_need_input=[]
                print(t_batch)
                for col in self.opt.inputs_cols:
                    if type(t_sample_batched[col]) != dict:
                        t_need_input.append(t_sample_batched[col].to(self.opt.device))
                    else:
                        # 将多个dependency_tool 放入
                        for key, value in t_sample_batched[col].items():
                            t_sample_batched[col][key] = value.to(self.opt.device)
                        t_need_input.append(t_sample_batched[col])
                text_indices, aspect_indices, left_indices, dependency_graph_undir, \
                dependency_edge_undir, position_syntax_indices, \
                aspect_double_idx, _ = t_need_input
                attention_mask = (text_indices != 0).long().to(self.opt.device)

                t_targets = t_sample_batched['polarity'].to(self.opt.device)

                outputs_all = self.model(
                    input_ids=text_indices,
                    graph_undir=dependency_graph_undir,
                    edge_matrix_undir=dependency_edge_undir,
                    position_syntax_indices=position_syntax_indices,
                    aspect_double_idx=aspect_double_idx,
                    attention_mask=attention_mask,
                    output_attentions = "Yes",


                )
                # print(outputs_all)
                t_outputs = outputs_all[1]
                batch_attention = None
                attentions = outputs_all.attentions
                # print(attentions)
                # print(attentions[11].shape)
                for i in range(len(attentions)):
                    # print(attentions[i].squeeze(-2).shape)
                    if batch_attention==None:
                        batch_attention=attentions[i].squeeze(-2).cpu()
                    else:
                        batch_attention = torch.cat((batch_attention,attentions[i].squeeze(-2).cpu()),dim=0)

                attention_all.append(batch_attention)




                # print(t_outputs)
                # print(t_targets)
                # print("--------------------------------------")
                # print(torch.argmax(t_outputs, -1).cpu())
                # print(t_outputs)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        # attention_output = open('./attention_view/Attention_EnsembleDB_rest15_7342.pkl', 'wb')
        attention_output = open('./attention_view/Attention_Ensemble_rest15_7331.pkl.pkl', 'wb')

        pickle.dump(attention_all, attention_output)
        # print(len(attention_all))
        # print(t_targets_all.shape)
        test_acc = n_test_correct / n_test_total
        test_loss_all /= n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                              average='macro')
        print(f1)
        exit()



    def run(self):
        # Loss and Optimizer
        path = './state_dict/8-19/_EnsembleDBspacy_only_rest15_7331.pkl'

        self.get_evaluate_attention(path)
        exit()
        criterion = nn.CrossEntropyLoss()

        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)


        time_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        f_out = codecs.open('log/' + self.opt.model_name + '_' + self.opt.dataset +'dependency_type'+str(self.opt.dependency_type)+ '_val.txt', 'a+',encoding="utf-8")
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        f_out.write('count:{0}_repeat:{1}_time:{2}\n'.format(self.opt.count,self.opt.repeat,time_str))
        arguments = " "
        for arg in vars(self.opt):
            if ((arg=="position_matrix")==True)|((arg=="dependency_matrix")==True):
                continue
            arguments += '{0}: {1} '.format(arg, getattr(self.opt, arg))
        f_out.write(arguments)
        # self._reset_params()
        # optimizer.param_groups[0]['lr'] = self.opt.learning_rate

        max_test_acc, max_test_f1 = self._train(criterion)


        print("----------------------------")
        print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
        print('#' * 100)
        f_out.write('max_test_acc: {0}, max_test_f1: {1}\n'.format(max_test_acc, max_test_f1))
        f_out.close()
        return  max_test_acc,max_test_f1

