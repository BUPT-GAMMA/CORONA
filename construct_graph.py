import pickle
import torch
import numpy as np
from torch_geometric.data import HeteroData, Data
from tqdm import tqdm
import os
import json
import random

data_path = os.environ.get('DATA_DIR', '')
dataset = os.environ.get('DATASET_DIR', '/netflix_data')
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
top_k = int(os.environ.get('TOP_K', '500'))
dataset_name = os.path.basename(dataset).lstrip('/')

def construct_ui_graph():
    retrieved_nodes = torch.load(f'{data_path}/Graph_RA_Rec/model_states/retrieved_nodes_{top_k}_all.pth')
    ui_graph = ui_graph_raw = pickle.load(open(data_path + dataset + '/train_mat','rb'))
    n_users = ui_graph.shape[0]
    n_items = ui_graph.shape[1]
    iu_graph = ui_graph.T
    ui_graph_dict = {}
    iu_graph_dict = {}
    items_connected_dict = {}
    total_items_length = 0
    for index, item in tqdm(retrieved_nodes.items()):
        item = item.cpu()
        retrieved_nodes_np = item.numpy()
        sub_ui_graph = ui_graph[retrieved_nodes_np, :]
        items_connected = sub_ui_graph.nonzero()[1]
        items_connected = np.unique(items_connected)
        sub_ui_graph = sub_ui_graph[:, items_connected]
        sub_iu_graph = sub_ui_graph.T
        ui_graph_dict[index] = sub_ui_graph
        iu_graph_dict[index] = sub_iu_graph
        items_connected_dict[index] = items_connected.tolist()
        total_items_length += len(items_connected)
        row, col = sub_ui_graph.nonzero()
        user_ids = retrieved_nodes_np[row]
        item_ids = col
        data = HeteroData()
        data['user'].num_nodes = n_users
        data['item'].num_nodes = n_items
        edge_index = torch.tensor([user_ids, item_ids], dtype=torch.long)
        data['user', 'interacts', 'item'].edge_index = edge_index
        graphs_dir = f'{data_path}/Graph_RA_Rec/{dataset_name}/graphs'
        os.makedirs(graphs_dir, exist_ok=True)
        torch.save(data, f'{graphs_dir}/{index}.pt')
    avg_items_length = total_items_length / len(retrieved_nodes)
    # pickle.dump(ui_graph_list, open(data_path + '/Graph_RA_Rec/retrieved_graph/ui_retrieved_68_3878.pkl', 'wb'))
    retrieved_dir = f'{data_path}/Graph_RA_Rec/{dataset_name}/retrieved_graph'
    os.makedirs(retrieved_dir, exist_ok=True)
    pickle.dump(items_connected_dict, open(f'{retrieved_dir}/items_retrieved_{top_k}.pkl', 'wb'))
    print('done')

def construct_user_graph():
    retrieved_nodes = torch.load(f'{data_path}/Graph_RA_Rec/model_states/retrieved_nodes_{top_k}_all.pth', map_location='cpu')
    ui_graph = pickle.load(open(data_path + dataset + '/train_mat','rb'))
    n_users = ui_graph.shape[0]
    n_items = ui_graph.shape[1]
    iu_graph = ui_graph.T
    # ui_graph.to('cuda')
    # iu_graph.to('cuda')
    user_user_matrix = ui_graph * iu_graph
    user_user_matrix.setdiag(0)
    user_user_matrix.eliminate_zeros()
    # user_user_matrix = user_user_matrix.to('cpu')
    all_retrieved_map = {}
    
    for index, item in tqdm(retrieved_nodes.items()):
        item = item.cpu()
        retrieved_nodes_np = item.numpy()
        sub_user_user_matrix = user_user_matrix[retrieved_nodes_np][:, retrieved_nodes_np]
        row, col = sub_user_user_matrix.nonzero()
        # original_row = retrieved_nodes_np[row]
        # original_col = retrieved_nodes_np[col]
        # edge_index = torch.tensor([original_row, original_col], dtype=torch.long)
        edge_index = torch.tensor([row, col], dtype=torch.long)
        data = Data(edge_index=edge_index)
        # data.num_nodes = n_users
        data.num_nodes = len(retrieved_nodes_np)
        node_idx_to_user_id = {idx: uid for idx, uid in enumerate(retrieved_nodes_np)}
        all_retrieved_map[index] = node_idx_to_user_id
        user_graph_dir = f'{data_path}/Graph_RA_Rec/{dataset_name}/user_graphs_batch_{top_k}'
        os.makedirs(user_graph_dir, exist_ok=True)
        torch.save(data, f'{user_graph_dir}/{index}.pt')
    pickle.dump(all_retrieved_map, open(f'{data_path}/Graph_RA_Rec/{dataset_name}/user_graphs_batch_{top_k}/all_retrieved_map.pkl', 'wb'))
    print('done')

construct_ui_graph()

# items_retrieved = pickle.load(open(data_path + '/Graph_RA_Rec/retrieved_graph/items_retrieved_70_3500_4229.pkl','rb'))
# train_file = data_path + dataset + '/train.json'
# val_file = data_path + dataset + '/val.json' 
# test_file = data_path + dataset + '/test.json'
# train = json.load(open(train_file))
# test = json.load(open(test_file))
# val = json.load(open(val_file))

# ui_graph = pickle.load(open(data_path + dataset + '/train_mat','rb'))
# iu_graph = ui_graph.T
# count = 0
# for uid, test_items in test.items():
#     if len(test_items) == 0:
#         continue
#     if iu_graph[test_items[0]].nnz == 0 and count < 40 and uid not in items_retrieved.keys():
#         candidate = random.sample(range(0, ui_graph.shape[1]), 4)
#         candidate.append(test_items[0])
#         items_retrieved[uid] = candidate
#         count += 1

# keys = list(items_retrieved.keys())
# random.shuffle(keys)
# shuffled_items_retrieved = {key: items_retrieved[key] for key in keys}
# print('done')
# pickle.dump(shuffled_items_retrieved, open(data_path + '/Graph_RA_Rec/retrieved_graph/items_retrieved_70_3500_4229.pkl','wb'))
