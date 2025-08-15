import torch
import torch.nn as nn
from model import NodeRetrieverModel
import pickle
import scipy.sparse as sp
import numpy as np
from collections import deque, defaultdict
from tqdm import tqdm
import json
import random
import os
from time import time

MAX_DISTANCE = 5

class Trainer(object):
    def __init__(self):
        self.data_path = os.environ.get('DATA_DIR', '.')
        self.dataset = os.environ.get('DATASET_DIR', '/netflix_data')
        self.top_k = 100
        self.epoch = 7
        self.non_node = 0
        self.train_data = json.load(open(self.data_path + self.dataset + '/train.json'))
        self.val_data = json.load(open(self.data_path + self.dataset + '/val.json'))
        self.test_data = json.load(open(self.data_path + self.dataset + '/test.json'))
        self.ui_graph = pickle.load(open(self.data_path + self.dataset + '/train_mat','rb'))
        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]        
        self.iu_graph = self.ui_graph.T
        self.ui_adj_list = self.csr_to_adj_list(self.ui_graph)
        self.iu_adj_list = self.csr_to_adj_list(self.iu_graph)
        self.ui_graph = self.csr_norm(self.ui_graph, mean_flag=True)
        self.iu_graph = self.csr_norm(self.iu_graph, mean_flag=True)
        self.ui_graph = self.matrix_to_tensor(self.ui_graph)
        self.iu_graph = self.matrix_to_tensor(self.iu_graph)
        # self.ui_graph = self.ui_graph.cuda()
        # self.iu_graph = self.iu_graph.cuda()
        # self.ui_graph = self.ui_graph.coalesce()
        # self.iu_graph = self.iu_graph.coalesce()
        self.user_init_embedding = torch.tensor(pickle.load(open(self.data_path + self.dataset + '/augmented_user_init_embedding_final','rb')))
        if self.user_init_embedding.dtype != torch.float32:
            self.user_init_embedding = self.user_init_embedding.float()
        self.embedding_dim = self.user_init_embedding.shape[1]
        self.user_init_embedding = self.user_init_embedding.cuda()
        
        # preprocess
        self.distance_matrix = [[MAX_DISTANCE] * self.n_users for _ in range(self.n_users)]
        

        self.model = NodeRetrieverModel(self.embedding_dim, self.top_k)
        self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def data_generator(self, uid, train_items):
        uid = int(uid)
        ui_adj_list = self.ui_adj_list
        iu_adj_list = self.iu_adj_list
        if len(train_items) == 0:
            return None, None, None
        if train_items[-1] in ui_adj_list[uid]:
            ui_adj_list[uid].remove(train_items[-1])
        if uid in iu_adj_list[train_items[-1]]:
            iu_adj_list[train_items[-1]].remove(uid)
        temp_ui_graph, temp_iu_graph = self.remove_edge_from_graph(uid, train_items[-1])
        distance_tensor = self.update_distance_tensor(temp_ui_graph, temp_iu_graph, uid)
        source_node_idx = uid   
        shortest_path = self.bfs_shortest_path(ui_adj_list, iu_adj_list, source_node_idx, train_items[-1])
        if shortest_path is not None:
            lable_node_idx = shortest_path[-2]
        else:
            # lable_node_idx = torch.randint(0, self.n_users, (1,)).item()
            self.non_node += 1
            print(f"Shortest path not found for user {uid} and item {train_items[-1]}, {self.non_node} in total.")
            return None, None, None
        return source_node_idx, lable_node_idx, distance_tensor
    
    def val_test_data_generator(self, uid, val_item):
        if val_item:
            uid = int(uid)
            ui_adj_list = self.ui_adj_list
            iu_adj_list = self.iu_adj_list
            distance_tensor = self.update_distance_tensor(self.ui_graph, self.iu_graph, uid)
            source_node_idx = uid
            shortest_path = self.bfs_shortest_path(ui_adj_list, iu_adj_list, source_node_idx, val_item)
            if shortest_path is not None:
                label_node_idx = shortest_path[-2]
            else:
                return None, None, None
            return source_node_idx, label_node_idx, distance_tensor
        else:
            return None, None, None
                
    
    def remove_edge_from_graph(self, uid, item):
        # 获取self.ui_graph的indices和values
        ui_indices = self.ui_graph.coalesce().indices()  # (2, nnz)
        ui_values = self.ui_graph.coalesce().values()    # (nnz)

        # 找到要删除的用户-物品连边索引
        mask_ui = ~((ui_indices[0] == uid) & (ui_indices[1] == item))

        # 构建新的self.ui_graph
        new_ui_indices = ui_indices[:, mask_ui]
        new_ui_values = ui_values[mask_ui]
        new_ui_graph = torch.sparse_coo_tensor(new_ui_indices, new_ui_values, self.ui_graph.size(), device=self.ui_graph.device)

        # 获取self.iu_graph的indices和values
        iu_indices = self.iu_graph.coalesce().indices()  # (2, nnz)
        iu_values = self.iu_graph.coalesce().values()    # (nnz)

        # 找到要删除的物品-用户连边索引
        mask_iu = ~((iu_indices[0] == item) & (iu_indices[1] == uid))

        # 构建新的self.iu_graph
        new_iu_indices = iu_indices[:, mask_iu]
        new_iu_values = iu_values[mask_iu]
        new_iu_graph = torch.sparse_coo_tensor(new_iu_indices, new_iu_values, self.iu_graph.size(), device=self.iu_graph.device)

        return new_ui_graph, new_iu_graph

    def find_neighboring_users_gpu(self, ui_graph, iu_graph, target_user):
        target_user_items = ui_graph[target_user].to_dense()  # 稀疏矩阵变为密集格式
        potential_users = iu_graph.t().mm(target_user_items.unsqueeze(1)).to_dense().squeeze()

        potential_users[target_user] = 0
        neighboring_users = potential_users.nonzero(as_tuple=True)[0]  # 返回索引张量
        
        return neighboring_users 


    def bfs_shortest_path(self, ui_adj_list, iu_adj_list, start_user, target_item):
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
    
    
    def update_distance_tensor(self, ui_graph, iu_graph, target_user):
        # 创建一个长度为 num_users 的张量，初始值为 MAX_DISTANCE
        num_users = ui_graph.size(0)
        distance_tensor = torch.full((num_users,), MAX_DISTANCE, device='cuda')

        # 获取 ui_graph 的非零索引和对应值
        ui_indices = ui_graph.coalesce().indices()  # (2, nnz)，其中第1行是用户，第2行是物品
        iu_indices = iu_graph.coalesce().indices()  # (2, nnz)，其中第1行是物品，第2行是用户

        # 找到与目标用户交互过的物品（即在 ui_graph 中目标用户作为起始节点的那些物品）
        first_hop_items = ui_indices[1][ui_indices[0] == target_user]  # 目标用户交互过的物品的索引

        # 找到与这些物品有交互的其他用户（这些用户就是一跳邻居）
        first_hop_users = iu_indices[1][iu_indices[0].unsqueeze(1).eq(first_hop_items).any(dim=1)]
        first_hop_users = first_hop_users[first_hop_users != target_user]  # 排除目标用户自己

        # 更新一跳邻居的距离为 1
        distance_tensor[first_hop_users] = 1

        # 找到二跳邻居：通过一跳邻居找到他们交互过的物品
        second_hop_items = ui_indices[1][ui_indices[0].unsqueeze(1).eq(first_hop_users).any(dim=1)]
        second_hop_users = iu_indices[1][iu_indices[0].unsqueeze(1).eq(second_hop_items).any(dim=1)]

        # 排除一跳邻居和目标用户
        second_hop_users = second_hop_users[~first_hop_users.unsqueeze(1).eq(second_hop_users).any(dim=0)]
        second_hop_users = second_hop_users[second_hop_users != target_user]

        # 更新二跳邻居的距离为 2
        distance_tensor[second_hop_users] = 2

        return distance_tensor



    def csr_norm(self, csr_mat, mean_flag=False):
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
        
    def matrix_to_tensor(self, cur_matrix):
            if type(cur_matrix) != sp.coo_matrix:
                cur_matrix = cur_matrix.tocoo()  #
            indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
            values = torch.from_numpy(cur_matrix.data)  #
            shape = torch.Size(cur_matrix.shape)
            return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()

    def csr_to_adj_list(self, csr_mat):
        adj_list = defaultdict(list)
        coo_mat = csr_mat.tocoo()
        for u, v in zip(coo_mat.row, coo_mat.col):
            adj_list[u].append(v)
        return adj_list
    
    def evaluate(self):
        self.model.eval()
        total_val_loss = 0.0
        onehop_count = 0
        with torch.no_grad():
            recall = 0
            if os.path.exists(self.data_path + self.dataset + '/val_for_RA.pkl'):
                print('Evaluating...')
                self.val_data = pickle.load(open(self.data_path + self.dataset + '/val_for_RA.pkl', 'rb'))
                for uid, (source_node_idx, label_node_idx, distance_tensor) in tqdm(self.val_data.items()):
                    distance_tensor = distance_tensor.cuda()
                    loss, top_k_indices = self.model(self.user_init_embedding, source_node_idx, distance_tensor, label_node_idx)
                    total_val_loss += loss.item()
                    neighboring_users = self.find_neighboring_users_gpu(self.ui_graph, self.iu_graph, source_node_idx)
                    if label_node_idx in neighboring_users:
                        onehop_count += 1
                    if not torch.all(torch.isin(neighboring_users, top_k_indices)):
                        print('debug')
                    if label_node_idx in top_k_indices:
                        recall += 1
            else:
                processed_val_data = {}
                print('Processed val data not found. Processing and Evaluating...')
                for uid, val_items in tqdm(self.val_data.items()):
                    source_node_idx, label_node_idx, distance_tensor = self.val_test_data_generator(uid, val_items)
                    if source_node_idx is None:
                        continue
                    else:
                        save_tensor = distance_tensor.cpu()
                        processed_val_data[uid] = (source_node_idx, label_node_idx, save_tensor)
                    loss, top_k_indices = self.model(self.user_init_embedding, source_node_idx, distance_tensor, label_node_idx)
                    if label_node_idx in top_k_indices:
                        recall += 1
                    total_val_loss += loss.item()
                pickle.dump(processed_val_data, open(self.data_path + self.dataset + '/val_for_RA.pkl', 'wb'))
        average_val_loss = total_val_loss / len(self.val_data)
        average_recall = recall / len(self.val_data)
        print(f'One-hop count: {onehop_count}')
        return average_val_loss, average_recall
    
    def test(self):
        best_model_state = torch.load(self.data_path + f'/Graph_RA_Rec/model_states/best_model_state_{self.top_k}.pth')
        self.model.load_state_dict(best_model_state)
        self.model.eval()
        total_test_loss = 0.0
        all_top_k_indices = {}
        
        with torch.no_grad():
            recall = 0
            if os.path.exists(self.data_path + self.dataset + '/test_for_RA.pkl'):
                print('Testing...')
                self.test_data = pickle.load(open(self.data_path + self.dataset + '/test_for_RA.pkl', 'rb'))
                for uid, (source_node_idx, label_node_idx, distance_tensor) in tqdm(self.test_data.items()):
                    distance_tensor = distance_tensor.cuda()
                    loss, top_k_indices = self.model(self.user_init_embedding, source_node_idx, distance_tensor, label_node_idx)
                    total_test_loss += loss.item()
                    all_top_k_indices[uid] = top_k_indices
                    if label_node_idx in top_k_indices:
                        recall += 1
            else:
                processed_test_data = {}
                print('Processed test data not found. Processing and Testing...')
                for uid, test_items in tqdm(self.test_data.items()):
                    source_node_idx, label_node_idx, distance_tensor = self.val_test_data_generator(uid, test_items)
                    if source_node_idx is None:
                        continue
                    else:
                        save_tensor = distance_tensor.cpu()
                        processed_test_data[uid] = (source_node_idx, label_node_idx, save_tensor)
                    loss, top_k_indices = self.model(self.user_init_embedding, source_node_idx, distance_tensor, label_node_idx)
                    total_test_loss += loss.item()
                    all_top_k_indices.append(top_k_indices)
                    if label_node_idx in top_k_indices:
                        recall += 1
                pickle.dump(processed_test_data, open(self.data_path + self.dataset + '/test_for_RA.pkl', 'wb'))
            self.train_data = pickle.load(open(self.data_path + self.dataset + '/train_for_RA.pkl', 'rb'))
            for uid, (source_node_idx, label_node_idx, distance_tensor) in tqdm(self.train_data.items()):
                distance_tensor = distance_tensor.cuda()
                loss, top_k_indices = self.model(self.user_init_embedding, source_node_idx, distance_tensor, label_node_idx)
                total_test_loss += loss.item()
                all_top_k_indices[uid] = top_k_indices
                if label_node_idx in top_k_indices:
                    recall += 1
        average_test_recall = recall / (len(self.test_data) + len(self.train_data))
        average_test_loss = total_test_loss / (len(self.test_data) + len(self.train_data))
        return average_test_loss, average_test_recall, all_top_k_indices
    
    def train(self):
        best_val_loss = float('inf')
        best_recall = 0
        for epoch in range(self.epoch):
            self.model.train()  # 设置模型为训练模式
            total_loss = 0.0  # 累积损失
            all_top_k_indices = []  
            train_recall = 0
            if os.path.exists(self.data_path + self.dataset + '/train_for_RA.pkl'):
                print('Training...')
                self.train_data = pickle.load(open(self.data_path + self.dataset + '/train_for_RA.pkl', 'rb'))
                for uid, (source_node_idx, label_node_idx, distance_tensor) in tqdm(self.train_data.items()):
                    distance_tensor = distance_tensor.cuda()
                    self.optimizer.zero_grad()
                    loss, top_k_indices = self.model(self.user_init_embedding, source_node_idx, distance_tensor, label_node_idx)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    all_top_k_indices.append(top_k_indices)
                    if label_node_idx in top_k_indices:
                        train_recall += 1
            else:
                print('Processed train data not found. Processing and Training...')
                processed_train_data = {}
                for uid, train_items in tqdm(self.train_data.items()):
                    t1 = time()
                    source_node_idx, label_node_idx, distance_tensor = self.data_generator(uid, train_items)
                    if source_node_idx is None:
                        continue
                    else:
                        save_tensor = distance_tensor.cpu()
                        processed_train_data[uid] = (source_node_idx, label_node_idx, save_tensor)
                    self.optimizer.zero_grad()  
                    t2 = time()
                    loss, top_k_indices = self.model(self.user_init_embedding, source_node_idx, distance_tensor, label_node_idx)
                    
                    # 反向传播和优化
                    loss.backward()
                    self.optimizer.step()
                    
                    # 累积损失
                    total_loss += loss.item()
                    # all_top_k_indices.append(top_k_indices)  # 记录每个样本的top-k结果
                    if label_node_idx in top_k_indices:
                        train_recall += 1
                    t3 = time()
                    # print(f'Data generate: {t2 - t1}, Model: {t3 - t2}.')
                pickle.dump(processed_train_data, open(self.data_path + self.dataset + '/train_for_RA.pkl', 'wb'))
            average_loss = total_loss / len(self.train_data)
            average_recall = train_recall / len(self.train_data)
            if self.val_data is not None:
                val_loss, val_recall = self.evaluate()
                if val_recall > best_recall:
                    best_recall = val_recall
                    best_model_state = self.model.state_dict()
                    print(f"Epoch [{epoch+1}/{self.epoch}], Train Loss: {average_loss:.4f}, Train Recall: {average_recall}, Val Loss: {val_loss:.4f}, Val Recall: {val_recall}.")
            else:
                print(f"Epoch [{epoch+1}/{self.epoch}], Trian Loss: {average_loss:.4f}, Train Recall: {average_recall}.")
        if self.val_data is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        os.makedirs(os.path.join(self.data_path, 'Graph_RA_Rec', 'model_states'), exist_ok=True)
        torch.save(best_model_state, self.data_path + f'/Graph_RA_Rec/model_states/best_model_state_{self.top_k}.pth')
        return average_loss, average_recall

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", '0')
    set_seed(3)
    trainer = Trainer()
    trainer.train()
    test_loss, test_recall, retrieved_nodes = trainer.test()
    os.makedirs(os.path.join(trainer.data_path, 'Graph_RA_Rec', 'model_states'), exist_ok=True)
    torch.save(retrieved_nodes, trainer.data_path + f'/Graph_RA_Rec/model_states/retrieved_nodes_{trainer.top_k}_all.pth')
    print(f"Test Loss: {test_loss}, Test Recall: {test_recall}")