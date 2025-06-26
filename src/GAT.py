import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def sample_neighbors(adj_list, nodes, num_neighbors):
    """
    为一批节点采样其邻居，构建子图。
    adj_list: dict {node: [neighbors]}
    nodes: list of target node indices
    num_neighbors: 每个节点采样的邻居数量
    返回: sampled_nodes (去重后), edge_index (2, E) 形式
    """
    sampled = set(nodes)
    edges = []
    for n in nodes:
        neigh = adj_list.get(n, [])
        if len(neigh) == 0:
            continue
        sampled_neigh = random.sample(neigh, min(len(neigh), num_neighbors))
        for nbr in sampled_neigh:
            edges.append((nbr, n))
        sampled.update(sampled_neigh)
    sampled_nodes = list(sampled)
    # 构造映射
    id_map = {nid: idx for idx, nid in enumerate(sampled_nodes)}
    # 构造edge_index
    edge_index = torch.LongTensor([[id_map[u] for u, v in edges],
                                   [id_map[v] for u, v in edges]])
    return sampled_nodes, edge_index


class GraphAttentionLayer(nn.Module):
    """
    单个图注意力层（GAT Layer）。
    输入：特征 h (N, in_feats) 以及边列表 edge_index (2, E)
    输出：新的特征 h' (N, out_feats)
    """
    def __init__(self, in_feats, out_feats, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(in_feats, out_feats))
        self.a = nn.Parameter(torch.empty(2 * out_feats, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index):
        Wh = torch.mm(h, self.W)  # (N, out_feats)
        N = Wh.size(0)
        # 计算注意力得分
        src, dst = edge_index
        Wh_src = Wh[src]  # (E, out_feats)
        Wh_dst = Wh[dst]
        a_input = torch.cat([Wh_src, Wh_dst], dim=1)  # (E, 2*out_feats)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))  # (E,)
        # 对于每个目标节点归一化
        attention = torch.zeros(edge_index.size(1), device=h.device)
        # 聚合归一化
        dst_unique, dst_counts = torch.unique(dst, return_counts=True)
        # 计算softmax per dst
        # 这里用指数映射再手动分段softmax
        exp_e = torch.exp(e)
        denom = torch.zeros_like(exp_e)
        # 累加每个 dst 的 exp_e
        denom_dict = {}
        for idx, d in enumerate(dst.tolist()):
            denom_dict.setdefault(d, 0.0)
            denom_dict[d] += exp_e[idx].item()
        for idx, d in enumerate(dst.tolist()):
            attention[idx] = exp_e[idx] / denom_dict[d]
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 信息传递
        h_prime = torch.zeros_like(Wh)
        for idx in range(edge_index.size(1)):
            h_prime[dst[idx]] += attention[idx] * Wh[src[idx]]
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GraphAttentionNetwork(nn.Module):
    """
    GAT，支持邻居采样的批训练。
    """
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, num_neighbors):
        super(GraphAttentionNetwork, self).__init__()
        self.dropout = dropout
        self.num_neighbors = num_neighbors
        # 多头第一层
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        # 输出层
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, concat=False)

    def forward(self, h, edge_index):
        x = F.dropout(h, self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, edge_index)
        return F.log_softmax(x, dim=1)

    def train_epoch(self, features, labels, adj_list, train_nodes, optimizer, batch_size):
        """
        单个 epoch 的批训练。
        features: 原始特征矩阵 (N, D)
        labels: 标签向量 (N,)
        adj_list: 邻接列表 dict
        train_nodes: 可训练节点列表
        optimizer: 优化器
        batch_size: 批大小
        """
        self.train()
        total_loss = 0.0
        random.shuffle(train_nodes)
        for i in range(0, len(train_nodes), batch_size):
            batch = train_nodes[i:i+batch_size]
            sampled_nodes, edge_index = sample_neighbors(adj_list, batch, self.num_neighbors)
            batch_idx = [sampled_nodes.index(n) for n in batch]
            h_sub = features[sampled_nodes]
            labels_sub = labels[sampled_nodes]

            optimizer.zero_grad()
            out = self.forward(h_sub, edge_index)
            loss = F.nll_loss(out[batch_idx], labels_sub[batch_idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        return total_loss / len(train_nodes)

    def test(self, features, labels, adj_list, test_nodes, batch_size):
        """
        批测试方法。
        返回：准确率
        """
        self.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(test_nodes), batch_size):
                batch = test_nodes[i:i+batch_size]
                sampled_nodes, edge_index = sample_neighbors(adj_list, batch, self.num_neighbors)
                batch_idx = [sampled_nodes.index(n) for n in batch]
                h_sub = features[sampled_nodes]
                labels_sub = labels[sampled_nodes]
                out = self.forward(h_sub, edge_index)
                pred = out[batch_idx].max(1)[1]
                correct += pred.eq(labels_sub[batch_idx]).sum().item()
        return correct / len(test_nodes)


# 生成示例数据
N, D, C = 20, 8, 2
features = torch.randn(N, D)
labels = torch.randint(0, C, (N,))
adj_list = {i: random.sample([j for j in range(N) if j != i], 3) for i in range(N)}
nodes = list(range(N))
random.shuffle(nodes)
train_nodes, test_nodes = nodes[:15], nodes[15:]

# 实例化模型与优化器
model = GraphAttentionNetwork(nfeat=D, nhid=4, nclass=C, dropout=0.5, alpha=0.2, nheads=2, num_neighbors=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练与测试
train_loss = model.train_epoch(features, labels, adj_list, train_nodes, optimizer, batch_size=8)
test_acc = model.test(features, labels, adj_list, test_nodes, batch_size=8)

print(f"Train Loss: {train_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
