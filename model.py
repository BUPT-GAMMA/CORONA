import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeRetrieverModel(nn.Module):
    def __init__(self, embedding_dim, top_k):
        super(NodeRetrieverModel, self).__init__()
        self.distance_encoding = nn.Embedding(3, 2)
        self.linear = nn.Linear(embedding_dim + 2, embedding_dim, bias=False)
        self.top_k = top_k

    def forward(self, node_embeddings, source_node_idx, distance_tensor, label_node_idx):
        distances = distance_tensor.long()
        distance_indices = torch.clamp(distances - 1, min=0, max=2)
        distance_encodings = self.distance_encoding(distance_indices)
        all_node_embeddings = node_embeddings
        all_node_embeddings_with_distance = torch.cat((all_node_embeddings, distance_encodings), dim=1)
        transformed_embeddings = self.linear(all_node_embeddings_with_distance)
        source_embedding = node_embeddings[source_node_idx]
        
        cosine_sim = F.cosine_similarity(source_embedding.unsqueeze(0), transformed_embeddings, dim=1)

        distances = distance_tensor
        distances = distances.to(cosine_sim.device)
        valid_indices = (distances <= 2).nonzero(as_tuple=True)[0]
        valid_cosine_sim = cosine_sim[valid_indices]
        valid_distances = distances[valid_indices]

        distance_weights = torch.ones_like(valid_distances, dtype=torch.float32, device=cosine_sim.device)
        distance_weights[valid_distances == 1] *= 1.5
        distance_weights[valid_distances == 2] *= 2.0
        weighted_similarity = valid_cosine_sim * distance_weights
        if len(weighted_similarity) < self.top_k:
            top_k_indices = torch.arange(len(valid_indices), device=valid_indices.device)
        else:
            _, top_k_indices = torch.topk(weighted_similarity, self.top_k)
        top_k_embeddings = transformed_embeddings[valid_indices[top_k_indices]]

        label_embedding = node_embeddings[label_node_idx]
        query_embedding = node_embeddings[source_node_idx]

        loss = self.compute_loss(query_embedding, label_node_idx, transformed_embeddings)

        source_node_idx_tensor = torch.tensor([source_node_idx], dtype=torch.long, device=cosine_sim.device)
        one_hop_indices = (valid_distances == 1).nonzero(as_tuple=True)[0]
        final_top_k = torch.unique(torch.cat([valid_indices[top_k_indices], valid_indices[one_hop_indices], source_node_idx_tensor]))
        return loss, final_top_k

    def compute_loss(self, EQ_1, label_node_idx, transformed_embeddings):
        label_embeddings = transformed_embeddings[label_node_idx]
        numerator = torch.exp(torch.matmul(label_embeddings, EQ_1))
        denominator = torch.exp(torch.matmul(transformed_embeddings, EQ_1)).sum()
        loss = -torch.log(numerator / denominator).sum()
        return loss

    
    # def compute_loss(self, top_k_embeddings, label_embedding):
    #     label_embedding = label_embedding.unsqueeze(0)
    #     similarities = F.cosine_similarity(top_k_embeddings, label_embedding, dim=1)  # (top_k,)
    #     similarity_scores = F.log_softmax(similarities, dim=0)  # (top_k,)
    #     loss = -similarity_scores.mean()
    #     return loss
    
    # def compute_loss(self, top_k_embeddings, all_node_embeddings, label_embedding, distances_to_topk):
    #     # Step 1: 计算top-k节点与label节点的cosine相似度
    #     similarities_top_k = F.cosine_similarity(top_k_embeddings, label_embedding.unsqueeze(0), dim=1)  # (top_k,)
        
    #     # Step 2: 计算所有节点与label节点的相似度，用于分母的归一化
    #     similarities_all = F.cosine_similarity(all_node_embeddings, label_embedding.unsqueeze(0), dim=1)  # (N,)

    #     # Step 3: 对相似度进行softmax归一化，用于计算top-k节点集中的概率
    #     softmax_denominator = torch.sum(torch.exp(similarities_all))  # 分母: sum over all nodes
    #     probabilities_top_k = torch.exp(similarities_top_k) / softmax_denominator  # (top_k,)

    #     # Step 4: 考虑距离进行加权，距离越近权重越大
    #     distance_weights = 1.0 / distances_to_topk.clamp(min=1.0)  # 防止除以0
    #     weighted_probabilities = probabilities_top_k * distance_weights  # (top_k,)
        
    #     # Step 5: 计算最终loss (取负对数)
    #     loss = -torch.log(weighted_probabilities).mean()

    #     return loss

