import numpy as np
import pickle
import scipy.sparse as sp
import torch
import json
from collections import deque, defaultdict
import sys

data_path = '/home/cjz'
dataset = '/netflix'
topk = 10

def csr_to_adj_list(csr_mat):
    adj_list = defaultdict(list)
    coo_mat = csr_mat.tocoo()
    for u, v in zip(coo_mat.row, coo_mat.col):
        adj_list[u].append(v)
    return adj_list

def find_neighboring_users(ui_adj_list, iu_adj_list, target_user):
    neighboring_users = set()
    target_items = ui_adj_list.get(target_user, [])
    for item in target_items:
        users_for_item = iu_adj_list.get(item, [])
        for user in users_for_item:
            if user != target_user:
                neighboring_users.add(user)
    
    return list(neighboring_users)


def bfs_shortest_path(ui_adj_list, iu_adj_list, start_user, target_item):
    
    # BFS 初始化
    queue = deque([(start_user, [start_user])])
    visited_users = set()
    visited_items = set()
    
    while queue:
        current_user, path = queue.popleft()
        
        if current_user in visited_users:
            continue
        
        visited_users.add(current_user)
        for item in ui_adj_list.get(current_user, []):
            if item == target_item:
                return path + [item]
            
            if item not in visited_items:
                visited_items.add(item)
                for user in iu_adj_list.get(item, []):
                    if user not in visited_users:
                        queue.append((user, path + [item, user]))
    
    return None 


def csr_norm(csr_mat, mean_flag=False):
    rowsum = np.array(csr_mat.sum(1))
    rowsum = np.power(rowsum+1e-8, -0.5).flatten()
    rowsum[np.isinf(rowsum)] = 0.
    rowsum_diag = sp.diags(rowsum)
    colsum = np.array(csr_mat.sum(0))
    colsum = np.power(colsum+1e-8, -0.5).flatten()
    colsum[np.isinf(colsum)] = 0.
    colsum_diag = sp.diags(colsum)
    if mean_flag == False:
        return rowsum_diag*csr_mat*colsum_diag
    else:
        return rowsum_diag*csr_mat
    
def matrix_to_tensor(cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()

image_feats = np.load(data_path + '{}/image_feat.npy'.format(dataset))
text_feats = np.load(data_path + '{}/text_feat.npy'.format(dataset))
image_feat_dim = image_feats.shape[-1]
text_feat_dim = text_feats.shape[-1]

ui_graph = ui_graph_raw = pickle.load(open(data_path + dataset + '/train_mat','rb'))

# get user embedding  
# augmented_user_init_embedding = pickle.load(open(data_path + dataset + '/augmented_user_init_embedding','rb'))
# augmented_user_init_embedding_list = []
# for i in range(len(augmented_user_init_embedding)):
#     augmented_user_init_embedding_list.append(augmented_user_init_embedding[i])
# augmented_user_init_embedding_final = np.array(augmented_user_init_embedding_list)
# pickle.dump(augmented_user_init_embedding_final, open(data_path + dataset + '/augmented_user_init_embedding_final','wb'))
user_init_embedding = pickle.load(open(data_path + dataset + '/augmented_user_init_embedding_final','rb'))

# get separate embedding matrix 
# augmented_total_embed_dict = {'year':[] , 'title':[], 'director':[], 'country':[], 'language':[]}
# augmented_atttribute_embedding_dict = pickle.load(open(data_path + dataset + '/augmented_atttribute_embedding_dict','rb'))
# for value in augmented_atttribute_embedding_dict.keys():
#     for i in range(len(augmented_atttribute_embedding_dict[value])):
#         augmented_total_embed_dict[value].append(augmented_atttribute_embedding_dict[value][i])   
#     augmented_total_embed_dict[value] = np.array(augmented_total_embed_dict[value])    
# pickle.dump(augmented_total_embed_dict, open(data_path + dataset + '/augmented_total_embed_dict','wb'))
item_attribute_embedding = pickle.load(open(data_path + dataset + '/augmented_total_embed_dict','rb'))  

item_embedding = (item_attribute_embedding['year'] + item_attribute_embedding['title'] + item_attribute_embedding['director'] + item_attribute_embedding['country'] + item_attribute_embedding['language']) / 5
# item_embedding = torch.from_numpy(item_embedding)
# topk_n_indices = torch.zeros(10, topk, dtype=torch.long)
# topk_n = torch.zeros(10, topk)
# # for i in range(0, 10):
#     q_emb = torch.from_numpy(user_init_embedding[i])
#     n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, item_embedding)
#     topk_n[i], topk_n_indices[i] = torch.topk(n_prizes, topk, largest=True)


image_ui_index = {'x':[], 'y':[]}
text_ui_index = {'x':[], 'y':[]}

n_users = ui_graph.shape[0]
n_items = ui_graph.shape[1]        
iu_graph = ui_graph.T

# ui_adj_list = csr_to_adj_list(ui_graph)
# iu_adj_list = csr_to_adj_list(iu_graph)
# ui_graph = csr_norm(ui_graph, mean_flag=True)
# iu_graph = csr_norm(iu_graph, mean_flag=True)
# ui_graph = matrix_to_tensor(ui_graph)
# iu_graph = matrix_to_tensor(iu_graph)

train_file = data_path + dataset + '/train.json'
val_file = data_path + dataset + '/val.json' 
test_file = data_path + dataset + '/test.json'
train = json.load(open(train_file))
test = json.load(open(test_file))
val = json.load(open(val_file))
cold_start = {}

for uid, test_items in test.items():
    if len(test_items) == 0:
        continue
    if iu_graph[test_items[0]].nnz <= 2:
        cold_start[uid] = test_items
# json.dump(cold_start, open(data_path + dataset + '/cold_start.json', 'w'))
cos_sim = torch.nn.CosineSimilarity(dim=-1)
# neighbor_prizes = cos_sim(torch.from_numpy(user_init_embedding[0]), item_embedding[adj_list[0]])
# sorted_sim, sorted_indices = torch.sort(neighbor_prizes, descending=True)

# path = bfs_shortest_path(ui_adj_list , iu_adj_list, 0, 14380) # 最短路    
augmented_attribute_dict = pickle.load(open(data_path + dataset + '/augmented_attribute_dict','rb'))

length_distribution = {}
# for key,value in val.items():
#     if value:
#         path = bfs_shortest_path(ui_adj_list , iu_adj_list, int(key), int(value[0]))
#         if path is not None:
#             path_length = len(path)
#             if path_length in length_distribution.keys():
#                 length_distribution[path_length] = length_distribution[path_length] + 1
#             else:
#                 length_distribution[path_length] = 1

# print("Path Length (including start user and target item) Distribution:", length_distribution)

# total_user_interactions = 0
# num_users = len(ui_adj_list)
# for user, items in ui_adj_list.items():
#     total_user_interactions += len(items)
# average_user_interactions = total_user_interactions / num_users if num_users > 0 else 0
# total_item_interactions = 0
# num_items = len(iu_adj_list)
# for item, users in iu_adj_list.items():
#     total_item_interactions += len(users)
# average_item_interactions = total_item_interactions / num_items if num_items > 0 else 0
if __name__ == "__main__":
    args = sys.argv
    source_node = int(args[1])
    print("Source Node:", source_node)
    print("Target Item:", test[str(source_node)])
    # target_item = train[str(source_node)][-1]
    # if target_item in ui_adj_list[source_node]:
    #     ui_adj_list[source_node].remove(target_item)
    # if source_node in iu_adj_list[target_item]:
    #     iu_adj_list[target_item].remove(source_node)
    # path = bfs_shortest_path(ui_adj_list , iu_adj_list, source_node, target_item) # 最短路
    # if path:
    #     print(path)
    # else:
    #     print("No path found.")
    