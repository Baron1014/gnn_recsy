from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *
import torch
import numpy as np
import pandas as pd

def save_txt(data_dict, file_name):
    out = []
    for key in data_dict.keys():
        out.append(data_dict[key])
        out[-1].insert(0, key)
        
    textfile = open("augmentation/{}.txt".format(file_name), "w")
    for element in out:
        s = ' '.join(element)
        textfile.write(s + "\n")
    textfile.close()

def getUserPosItems(users, UserItemNet):
        posItems = []
        for user in users:
            posItems.append(np.array(UserItemNet[user]))
        return posItems

def get_hot_items(allPos, hot=23):
    item_count = dict()
    for i in range(len(allPos)):
        pos_items = allPos[i]
        for j in pos_items:
            if item_count.get(j):
                item_count[j] += 1
            else:
                item_count[j] = 1
    print("total item: {}".format(len(item_count.keys())))
    print('max count: {}'.format(max(list(item_count.values()))))
    print('min count: {}'.format(min(list(item_count.values()))))
    item_df = pd.DataFrame({'item': item_count.keys(), 'count':list(item_count.values())})

    return item_df[item_df['count'] > hot]['item'].values

def user_item():
    args.device = torch.device('cpu')
    args.mess_dropout = eval(args.mess_dropout)
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    Recmodel = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    Recmodel = Recmodel.to(args.device)
    weight_file = "./model/NGCF_movielens.pth.tar"
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    print(f"loaded model weights from {weight_file}")

    allPos = data_generator.train_items
    hotItem_ratio = 65
    HotItems = get_hot_items(allPos, hot=hotItem_ratio)
    weak_num = 25
    weak_users = [i for i in range(len(allPos)) if len(allPos[i]) <= weak_num]
    # weak_num = 25
    # weak_users = [i for i in range(len(allPos))]
    batch_users_gpu = torch.Tensor(weak_users).long()
    batch_users_gpu = batch_users_gpu.to(args.device)
    # rating = Recmodel.getUsersRating(batch_users_gpu)
    item_batch = range(0, data_generator.n_items)
    u_g_embeddings, pos_i_g_embeddings, _ = Recmodel(batch_users_gpu,
                                                item_batch,
                                                [],
                                                drop_flag=False)
    rating = Recmodel.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
    allPos = getUserPosItems(weak_users, allPos)

    exclude_index = []
    exclude_items = []
    for range_i, items in enumerate(allPos):
        exclude_index.extend([range_i] * len(items))
        exclude_items.extend(items)
    rating[exclude_index, exclude_items] = -(1<<10)
    augmentation_dict = {}
    # rating = rating.detach().numpy() 
    top_k = 10
    if hotItem_ratio:
        des = 'hotItem{}_{}'.format(hotItem_ratio,top_k)
        _, rating_sort = torch.sort(rating, descending=True)
        rating_sort = rating_sort.numpy()
        for u, element in enumerate(rating_sort):
            user = str(weak_users[u])
            items = []
            for i in element:
                if i in HotItems:
                    items.append(str(i))
                    if len(items) == top_k:
                        break
            if len(items) != 0:
                augmentation_dict[user] = items
    else:
        des = top_k
        _, rating_K = torch.topk(rating, k=top_k)
        rating_K = rating_K.numpy()
        for u, element in enumerate(rating_K):
            user = str(weak_users[u])
            items = [str(i) for i in element]
            augmentation_dict[user] = items

    # rating_index = np.argwhere(rating>0.99)
    # for u, i in rating_index:
    #     user = str(weak_users[u])
    #     item = str(i)
    #     if augmentation_dict.get(user):
    #         augmentation_dict[user].append(item)
    #     else:
    #         augmentation_dict[user] = [item] 
    # min = 9999
    # max = 0
    # for k in augmentation_dict.keys():
    #     count = len(augmentation_dict[k])
    #     if  count > max:
    #         max = count
    #     if count < min:
    #         min = count
    # print("max number: {}, min number: {}".format(max, min))

    save_txt(augmentation_dict, "NGCF_aug_{}_{}".format(weak_num, des))

def user_user():
    args.device = torch.device('cpu')
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    Recmodel = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args).to(args.device)

    Recmodel = Recmodel.to(args.device)
    weight_file = "./model/NGCF_movielens.pth.tar"
    Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
    user_emb = Recmodel.get_user_embedding()
    print(f"loaded model weights from {weight_file}")

    # weak users
    # weak_num = 51
    allPos = data_generator.train_items
    # weak_users = [i for i in range(len(allPos)) if len(allPos[i]) <= weak_num]
    weak_num = 'all'
    weak_users = [i for i in range(len(allPos))]
    batch_users_gpu = torch.Tensor(weak_users).long()
    batch_users_gpu = batch_users_gpu.to(args.device)
    top_k = 3
    rating_K = Recmodel.getCosUserTop(batch_users_gpu, user_emb, k=top_k)

    augmentation_dict = {}
    # rating = rating.detach().numpy() 
    rating_K = rating_K.numpy()
    for u, element in enumerate(rating_K):
        user = str(weak_users[u])
        sim_user = [str(i) for i in element]
        augmentation_dict[user] = sim_user

    # rating_index = np.argwhere(rating>0.99)
    # for u, i in rating_index:
    #     user = str(weak_users[u])
    #     item = str(i)
    #     if augmentation_dict.get(user):
    #         augmentation_dict[user].append(item)
    #     else:
    #         augmentation_dict[user] = [item] 
    # min = 9999
    # max = 0
    # for k in augmentation_dict.keys():
    #     count = len(augmentation_dict[k])
    #     if  count > max:
    #         max = count
    #     if count < min:
    #         min = count
    # print("max number: {}, min number: {}".format(max, min))

    save_txt(augmentation_dict, "NGCF_SimUser_{}_{}".format(weak_num, top_k))



if __name__=="__main__":
    user_item()
    # user_user()