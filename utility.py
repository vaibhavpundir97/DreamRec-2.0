import os
import numpy as np
import pandas as pd
import random
import torch
import time as Time


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


def calculate_hit(sorted_list, topk, true_items, hit_purchase, ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def evaluate(model, test_data, diff, device, steps=1):
    t0 = Time.time()
    df = pd.read_pickle(test_data)
    batch_size = 100
    hit = [0]*3
    ndcg = [0]*3
    total = 0
    topk = [10, 20, 50]
    seqs, lens, tgt = list(df['seq'].values), list(
        df['len_seq'].values), list(df['next'].values)
    for i in range(len(seqs)//batch_size):
        bs = torch.LongTensor(
            np.array(seqs[i*batch_size:(i+1)*batch_size], dtype=np.int64)).to(device)
        bl = torch.tensor(lens[i*batch_size:(i+1)*batch_size],
                          dtype=torch.long, device=device)
        bt = torch.LongTensor(
            np.array(tgt[i*batch_size:(i+1)*batch_size], dtype=np.int64)).to(device)
        # states = torch.LongTensor(np.array(bs, dtype=np.int64)).to(device)

        if model.use_faiss and model.faiss_index is None:
            model.build_faiss_index()

        scores = model.predict(bs, bl, diff, steps)
        _, top_idx = scores.topk(100, dim=1)
        arr = top_idx.cpu().numpy()
        sorted_arr = np.flip(arr, axis=1).copy()
        calculate_hit(sorted_arr, topk, bt, hit, ndcg)
        total += batch_size
    print(f"HR@10 NDCG@10 HR@20 NDCG@20 HR@50 NDCG@50")
    metrics = []
    for idx in range(len(topk)):
        metrics.append(hit[idx]/total)
        metrics.append(ndcg[idx]/total)
    metrics = [m.item() if isinstance(m, torch.Tensor) else m for m in metrics]
    print(" ".join(f"{m:.4f}" for m in metrics))
    print("Evaluation Cost: " + Time.strftime("%H: %M: %S",
                                              Time.gmtime(Time.time() - t0)))
    return hit[1]/total  # HR@20


def save_model(model, dir, name):
    os.makedirs(dir, exist_ok=True)
    path = os.path.join(dir, name)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
