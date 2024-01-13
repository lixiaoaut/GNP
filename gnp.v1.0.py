#!/usr/bin/env python3
# ============================================================================== #
# Graph of Now and Past (GNP) @ PYG
# Powered by xiaolis@outlook.com 202401
# ============================================================================== #
# pip install pytorch torch_geometric pandas
cache_path = '/tmp/'
# ============================================================================== #
import os, math, torch
from torch import nn
from copy import deepcopy
from pandas import read_csv
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import scatter, softmax
from torch_geometric.typing import Tensor, PairTensor, SparseTensor
from torch_geometric.data import TemporalData, InMemoryDataset, download_url

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ============================================================================== #
class TDataset(InMemoryDataset):

    url = 'http://snap.stanford.edu/jodie/{}.csv'
    def __repr__(self): return self.name.capitalize() + '()'
    def __init__(self, root, name):
        self.name = name; super().__init__(root); self.load(self.processed_paths[0], data_cls=TemporalData)

    @property
    def raw_dir(self): return os.path.join(self.root, self.name, 'raw')
    def download(self): download_url(self.url.format(self.name), self.raw_dir)

    @property
    def processed_dir(self): return os.path.join(self.root, self.name, 'processed')
    @property
    def raw_file_names(self): return self.name + '.csv'

    @property
    def processed_file_names(self): return 'data.pt'
    def process(self):
        df = read_csv(self.raw_paths[0], skiprows=1, header=None)
        s, d, t, y = [torch.from_numpy(df.iloc[:,i].values).to(torch.int64) for i in range(4)]
        msg = torch.from_numpy(df.iloc[:,4:].values).to(torch.int64)
        d += int(s.max()) + 1; data = TemporalData(src=s, dst=d, t=t, msg=msg, y=y)
        self.save([data], self.processed_paths[0])

class TDataLoader(torch.utils.data.DataLoader):

    def __init__(self, data, batch_size=1, neg_sampling_ratio=0.):
        self.data, self.events_per_batch, self.neg_sampling_ratio = data, batch_size, neg_sampling_ratio
        if neg_sampling_ratio>0: self.min_dst, self.max_dst = int(data.dst.min()), int(data.dst.max())
        super().__init__(range(0, len(data), batch_size), 1, shuffle=False, collate_fn=self)

    def __call__(self, arange):
        batch = self.data[arange[0]: arange[0]+self.events_per_batch]; nids = [batch.src, batch.dst]
        if self.neg_sampling_ratio > 0:
            batch.neg_dst = torch.randint( low = self.min_dst, high = self.max_dst + 1,
                                           size = (round(self.neg_sampling_ratio*batch.dst.size(0)),),
                                           dtype = batch.dst.dtype, device = batch.dst.device )
            nids += [batch.neg_dst]
        batch.n_id = torch.cat(nids, dim=0).unique()
        return batch

# ============================================================================== #
class NeighborFinder:

    def __init__(self, n_node, size):
        self.eid = torch.empty((n_node, size), dtype = torch.int64, device = DEVICE)
        self._assoc = torch.empty (n_node, dtype = torch.int64, device = DEVICE)
        self.size = size; self.neighbors = self.eid.clone(); self.reset_state()

    def __call__(self, nid):
        nbrs, nodes, eid = self.neighbors[nid], nid.view(-1,1).repeat(1,self.size), self.eid[nid]
        mask = eid>=0; nbrs, nodes, eid = nbrs[mask], nodes[mask], eid[mask]
        nid = torch.cat([nid, nbrs]).unique()
        self._assoc[nid] = torch.arange(nid.size(0), device=nid.device)
        return nid, torch.stack([self._assoc[nbrs], self._assoc[nodes]]), eid

    def reset_state(self): self.cur_eid = 0; self.eid.fill_(-1)
    def insert(self, src, dst):
        nbrs, nodes = [torch.cat(t,dim=0) for t in [[src,dst],[dst,src]]]
        eid = torch.arange(self.cur_eid, self.cur_eid+src.size(0), device=src.device).repeat(2)
        self.cur_eid += src.numel(); nodes, perm = nodes.sort()
        nbrs, eid, nid = nbrs[perm], eid[perm], nodes.unique()
        self._assoc[nid] = torch.arange(nid.numel(), device=nid.device)
        d_id = torch.arange(nodes.size(0), device=nodes.device)%self.size + self._assoc[nodes].mul_(self.size)
        d_eid = eid.new_full((nid.numel() * self.size, ), -1); d_eid[d_id] = eid
        d_nbrs = eid.new_empty(nid.numel()*self.size); d_nbrs[d_id] = nbrs
        d_eid, d_nbrs = [x.view(-1, self.size) for x in [d_eid, d_nbrs]]
        nbrs = torch.cat([self.neighbors[nid, :self.size], d_nbrs], dim=-1)
        eid, perm = torch.cat([self.eid[nid, :self.size], d_eid], dim=-1).topk(self.size, dim=-1)
        self.eid[nid], self.neighbors[nid] = eid, torch.gather(nbrs,1,perm)

class TimeEncoder(nn.Module):

    def __init__(self, out_dim): super().__init__(); self.out_channels, self.lin = out_dim, nn.Linear(1, out_dim)
    def forward(self, t): return self.lin(t.view(-1,1)).cos()
    def reset_parameters(self): self.lin.reset_parameters()

class Message(nn.Module):

    def forward(self, src, dst, raw_msg, time_enc): return torch.cat([src, dst, raw_msg, time_enc], dim=-1)
    def __init__(self, raw_msg_dim, memory_dim, time_dim):
        super().__init__(); self.out_channels = raw_msg_dim + memory_dim*2 + time_dim

class Aggregator(nn.Module):

    def forward(self, msg, index, t, dim_size):
        argmax = self.scatter_argmax(t, index, dim=0, dim_size=dim_size); mask = argmax < msg.size(0)
        out = msg.new_zeros((dim_size, msg.size(-1))); out[mask] = msg[argmax[mask]]; return out

    def scatter_argmax(self, src, index, dim=0, dim_size=None):
        assert (src.dim() == 1 and index.dim() == 1) and (dim == 0 or dim == -1) and (src.numel() == index.numel())
        if dim_size is None: dim_size = index.max()+1 if index.numel()>0 else 0
        res = src.new_empty(dim_size); res.scatter_reduce_(0, index, src.detach(), reduce='amax', include_self=False)
        out = index.new_full([dim_size], fill_value=dim_size-1); nonzero = (src==res[index]).nonzero().view(-1)
        out[index[nonzero]] = nonzero; return out

class Memory(nn.Module):

    def __init__(self, n_node, raw_msg_dim, memory_dim, time_dim, msg, agg):
        super().__init__(); self.raw_msg_dim, self.num_nodes, self.time_dim = raw_msg_dim, n_node, time_dim
        self.msg_s_module, self.msg_d_module, self.agg, self.time_enc = msg, deepcopy(msg), agg, TimeEncoder(time_dim)
        self.memory, self.gru = torch.empty(n_node, memory_dim), nn.GRUCell(msg.out_channels, memory_dim)
        self.last_update = torch.empty(n_node, dtype=torch.int64); self._assoc = deepcopy(self.last_update)
        self.msg_s_store, self.msg_d_store = {}, {}; self.reset_parameters()

    @property
    def device(self): return self.time_enc.lin.weight.device
    def reset_state(self): zeros(self.memory); zeros(self.last_update); self._reset_msg_store()

    def detach(self): self.memory.detach_()
    def forward(self, nid): return self._get_updated_memory(nid) if self.training else (self.memory[nid], self.last_update[nid])

    def reset_parameters(self):
        modules = [self.msg_s_module, self.msg_d_module, self.agg, self.time_enc, self.gru]
        [m.reset_parameters() for m in modules if hasattr(m, 'reset_parameters')]; self.reset_state()

    def update_state(self, src, dst, t, raw_msg):
        nid = torch.cat([src,dst]).unique()
        if self.training : self._update_memory(nid)
        self._update_msg_store(src, dst, t, raw_msg, self.msg_s_store)
        self._update_msg_store(dst, src, t, raw_msg, self.msg_d_store)
        if not self.training: self._update_memory(nid)

    def train(self, mode=True):
        if ~self.training or ~mode: super().train(mode); return
        self._update_memory(torch.arange(self.num_nodes, device=self.device)); self._reset_msg_store(); super().train(mode)

    def _reset_msg_store(self):
        i = self.memory.new_empty((0,), device=self.device, dtype=torch.int64)
        msg = self.memory.new_empty((0, self.raw_msg_dim), device=self.device)
        self.msg_s_store = {j: (i,i,i,msg) for j in range(self.num_nodes)}; self.msg_d_store = deepcopy(self.msg_s_store)

    def _update_memory(self, nid): self.memory[nid], self.last_update[nid] = self._get_updated_memory(nid)
    def _update_msg_store(self, src, dst, t, raw_msg, msg_store):
        nid, perm = src.sort(); nid, count = nid.unique_consecutive(return_counts=True)
        for i, ix in zip(nid.tolist(), perm.split(count.tolist())): msg_store[i]=(src[ix],dst[ix],t[ix],raw_msg[ix])

    def _get_updated_memory(self, nid):
        self._assoc[nid] = torch.arange(nid.size(0), device=nid.device)
        msg_s, t_s, src_s, dst_s = self._compute_msg(nid, self.msg_s_store, self.msg_s_module)
        msg_d, t_d, src_d, dst_d = self._compute_msg(nid, self.msg_d_store, self.msg_d_module)
        idx, msg, t = [torch.cat(items, dim=0) for items in ([src_s, src_d], [msg_s, msg_d], [t_s, t_d])]
        memory = self.gru(self.agg(msg, self._assoc[idx], t, nid.size(0)), self.memory[nid])
        last_update = scatter(t, idx, 0, self.last_update.size(0), reduce='max')[nid]; return memory, last_update

    def _compute_msg(self, nid, msg_store, msg_module):
        data = [msg_store[i] for i in nid.tolist()]
        src, dst, t, raw_msg = list(zip(*[msg_store[i] for i in nid.tolist()]))
        src, dst, t, raw_msg = [torch.cat(x, dim=0) for x in [src, dst, t, raw_msg]]
        t_enc = self.time_enc((t-self.last_update[src]).to(raw_msg.dtype))
        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc); return msg, t, src, dst

class PastGraph(MessagePassing):

    def __init__(self, in_dim, out_dim, heads, dropout, edge_dim, bias=True, beta=False, concat=True, root_weight=True):
        super().__init__(node_dim=0); self.in_dim, self.out_dim = in_dim, out_dim
        self.heads, self.dropout, self.edge_dim, self.bias, self.concat = heads, dropout, edge_dim, bias, concat
        self._alpha = None; self.root_weight, self.beta = root_weight, beta and root_weight
        hdim = heads * out_dim; out_dim = hdim if concat else out_dim
        self.Q = Linear(in_dim, hdim); self.K, self.V = deepcopy(self.Q), deepcopy(self.Q)
        self.lin_skip = Linear(in_dim, out_dim, bias=bias)
        self.lin_edge, self.lin_beta = [Linear(i,o, bias=False) for i,o in [(edge_dim, hdim), (out_dim*3, 1)]]
        self.reset_parameters()

    def forward(self, x, edge_idx, edge_attr=None, return_attention_weights=None):
        if isinstance(x, Tensor): x: PairTensor = (x, x)
        q, k, v = [A.view(-1, self.heads, self.out_dim) for A in [self.Q(x[1]), self.K(x[0]), self.V(x[0])]]
        alpha = self._alpha; self._alpha = None
        out = self.propagate(edge_idx, query=q, key=k, value=v, edge_attr=edge_attr, size=None)
        out = out.view(-1, self.heads * self.out_dim) if self.concat else out.mean(dim=1)
        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None: beta = self.lin_beta(torch.cat([out, x_r, out-x_r], dim=-1)).sigmoid()
            out = out + x_r if self.lin_beta is None else beta * x_r + (1-beta) * out
        if not isinstance(return_attention_weights, bool): return out
        assert alpha is not None
        if isinstance(edge_idx, Tensor): return out, (edge_idx, alpha)
        if isinstance(edge_idx, SparseTensor): return out, edge_idx.set_value(alpha, layout='coo')

    def __repr__(self): return (f'{self.__class__.__name__}({self.in_dim}, {self.out_dim}, heads={self.heads})')
    def reset_parameters(self):
        super().reset_parameters()
        [obj.reset_parameters() for obj in [self.Q, self.K, self.V, self.lin_skip]]
        if self.edge_dim: self.lin_edge.reset_parameters()
        if self.beta: self.lin_beta.reset_parameters()

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i=0):
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_dim)
            key_j = key_j + edge_attr
        alpha = softmax((query_i * key_j).sum(dim=-1) / math.sqrt(self.out_dim), index, ptr, size_i)
        self._alpha = alpha
        alpha, out = nn.functional.dropout(alpha, p=self.dropout, training=self.training), value_j
        out = out + edge_attr if edge_attr is not None else out
        return out * alpha.view(-1, self.heads, 1)

class Now(nn.Module):

    def forward(self, x): return torch.cat([self.LA(x), self.LB(x)], dim=1)
    def __init__(self, in_dim, out_dim):
        super().__init__(); hdim = in_dim + in_dim
        self.LA = nn.Sequential( nn.Linear(in_dim, hdim), nn.BatchNorm1d(hdim),
                                 nn.Linear(hdim, int(out_dim*0.5)), nn.ReLU(inplace=True) )
        self.LB = nn.Sequential( nn.Linear(in_dim, hdim), nn.LayerNorm(hdim),
                                 nn.Linear(hdim, int(out_dim*0.5)), nn.ReLU(inplace=True) )
        [nn.init.xavier_uniform_(l.weight) for l in self.LA if isinstance(l, nn.Linear)]
        [nn.init.xavier_uniform_(l.weight) for l in self.LB if isinstance(l, nn.Linear)]

class GNP(nn.Module):

    def __init__(self, in_dim, out_dim, msg_dim, time_enc, mode=None):
        super().__init__(); self.mode = mode
        self.time_enc = time_enc; edim = msg_dim+time_enc.out_channels
        self.attn = PastGraph(in_dim=in_dim, out_dim=out_dim//2, heads=2, dropout=.1, edge_dim=edim)
        self.output = nn.Linear(out_dim+out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.now = Now(in_dim=msg_dim, out_dim=out_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        if self.mode=='now': return self.now(x)
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        past = self.attn(x, edge_index, edge_attr)
        if self.mode=='past': return past
        else: return self.output(torch.cat([self.norm(past), self.now(x)], dim=-1))

class EdgeTask(nn.Module):

    def __init__(self, i): super().__init__(); self.A, self.B, self.C = [nn.Linear(i,h) for i,h in [(i,i),(i,i),(i,1)]]
    def forward(self, src, dst): return self.C((self.A(src)+self.B(dst)).relu())

class Pack:

    def __init__(self, data, neighbor_loader, ablation):
        td = emb_dim = 100; nd = data.num_nodes; md = mmd = data.msg.size(-1)
        self.data, self.neighbor_loader = data, neighbor_loader
        self.memory = Memory(nd, md, mmd, td, Message(md, mmd, td), Aggregator()).to(DEVICE)
        self.gnn = GNP(mmd, emb_dim, md, self.memory.time_enc, ablation).to(DEVICE)
        self.pred = EdgeTask(emb_dim).to(DEVICE)
        self.assoc = torch.empty(data.num_nodes, dtype=torch.int64, device=DEVICE)

    def eval(self): {x.eval() for x in [self.memory, self.gnn, self.pred]}
    def train(self):
        {x.train() for x in [self.memory, self.gnn, self.pred]}
        {x.reset_state() for x in [self.memory, self.neighbor_loader]}
        return torch.optim.Adam(set().union(*(m.parameters() for m in [self.memory, self.gnn, self.pred])), lr=5e-4)

# ============================================================================== #
class Study:

    def __init__(self, dataset, ablation):
        print(f'GNP traning @{dataset} starting ...')
        data, trn_n_events, trn_loader, val_loader, tst_loader, neighbor_loader = self.get_dataloaders(dataset)
        pack = Pack(data, neighbor_loader, ablation); criterion = nn.BCEWithLogitsLoss()
        for epoch in range(10):
            loss = self.train(trn_loader, trn_n_events, pack, criterion); vap, vauc = self.eval(val_loader, pack)
            print(f'Epoch{epoch+1:02d}, Training Loss:{loss:.4f}, Validation AP:{vap:.4f} & ROC AUC: {vauc:.4f}')
        tap, tauc = self.eval(tst_loader, pack); print(f'Test AP: {tap:.4f} & ROC AUC:{tauc:.4f}')

    def get_dataloaders(self, name):
        dataset = TDataset(cache_path, name); data = dataset[0].to(DEVICE)
        tvt = data.train_val_test_split(val_ratio=.15, test_ratio=.15)
        trn_loader, val_loader, tst_loader = [TDataLoader(d, batch_size=100, neg_sampling_ratio=1.0) for d in tvt]
        neighbor_loader = NeighborFinder(data.num_nodes, size=20)
        return data, tvt[0].num_events, trn_loader, val_loader, tst_loader, neighbor_loader

    def train(self, trn_loader, trn_n_events, pack, criterion):
        total_loss, asc, optimizer = 0, pack.assoc, pack.train()
        for bth in trn_loader:
            optimizer.zero_grad(); bth = bth.to(DEVICE)
            nid, eix, eid = pack.neighbor_loader(bth.n_id)
            asc[nid] = torch.arange(nid.size(0), device=DEVICE)
            z, last_update = pack.memory(nid)
            z = pack.gnn(z, last_update, eix, pack.data.t[eid].to(DEVICE), pack.data.msg[eid].to(DEVICE))
            p, n = [pack.pred(z[asc[s]], z[asc[d]]) for s,d in [(bth.src, bth.dst),(bth.src, bth.neg_dst)]]
            loss = criterion(p, torch.ones_like(p)) + criterion(n, torch.zeros_like(n))
            pack.memory.update_state(bth.src, bth.dst, bth.t, bth.msg)
            pack.neighbor_loader.insert(bth.src, bth.dst)
            loss.backward(); optimizer.step(); pack.memory.detach()
            total_loss += float(loss) * bth.num_events
        return total_loss / trn_n_events

    @torch.no_grad()
    def eval(self, loader, pack):
        torch.manual_seed(42)
        asoc, aps, aucs = pack.assoc, [], []
        for bth in loader:
            bth = bth.to(DEVICE)
            nid, eix, eid = pack.neighbor_loader(bth.n_id)
            asoc[nid] = torch.arange(nid.size(0), device=DEVICE)
            z, last_update = pack.memory(nid)
            z = pack.gnn(z, last_update, eix, pack.data.t[eid].to(DEVICE), pack.data.msg[eid].to(DEVICE))
            pos = pack.pred(z[asoc[bth.src]], z[asoc[bth.dst]])
            neg = pack.pred(z[asoc[bth.src]], z[asoc[bth.neg_dst]])
            prd = torch.cat([pos, neg], dim=0).sigmoid().cpu()
            act = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))], dim=0)
            aps.append(average_precision_score(act, prd))
            aucs.append(roc_auc_score(act, prd))
            pack.memory.update_state(bth.src, bth.dst, bth.t, bth.msg)
            pack.neighbor_loader.insert(bth.src, bth.dst)
        return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())

# ============================================================================== #
if __name__ == '__main__':
    for run in range(10):
        print(f'\n\n--- Running {run} ---\n')
        [Study(d, 'now') for d in ['wikipedia', 'reddit', 'mooc', 'lastfm']]
        [Study(d, 'past') for d in ['wikipedia', 'reddit', 'mooc', 'lastfm']]
        [Study(d, 'both') for d in ['wikipedia', 'reddit', 'mooc', 'lastfm']]

# ============================================================================== #
