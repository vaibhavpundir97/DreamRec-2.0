import argparse
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utility import evaluate, save_model, setup_seed
from tenc import Tenc
from diffusion import Consistency

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DreamRec with Consistency Model Training.")
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of max epochs.')
    parser.add_argument('--random_seed', type=int,
                        default=100, help='Random seed')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='Batch size')
    parser.add_argument('--layers', type=int, default=1, help='GRU layers')
    parser.add_argument('--hidden_factor', type=int,
                        default=64, help='Embedding size')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='Noise levels for consistency training')
    parser.add_argument('--beta_start', type=float,
                        default=0.0001, help='Beta start for schedule')
    parser.add_argument('--beta_end', type=float,
                        default=0.02, help='Beta end for schedule')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--l2_decay', type=float,
                        default=0.0, help='L2 regularization')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device')
    parser.add_argument('--dropout_rate', type=float,
                        default=0.1, help='Dropout rate')
    parser.add_argument('--w', type=float, default=2.0,
                        help='Guidance strength')
    parser.add_argument('--p', type=float, default=0.1,
                        help='Masking probability')
    parser.add_argument(
        '--beta_sche', choices=['linear', 'exp', 'cosine'], default='exp', help='Beta schedule')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='Type of diffuser MLP: mlp1 or mlp2')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer: adam, adamw, adagrad, rmsprop')
    parser.add_argument('--save_model_dir', type=str,
                        default='ckpt', help='Path to save model checkpoint')
    parser.add_argument('--load_model_num', type=int,
                        default=0, help='load model checkpoint')
    parser.add_argument('--eval', action='store_true',
                        default=False, help='evaluate the model')
    return parser.parse_args()


def main(args):
    setup_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    data_directory = './data/yc'

    stats = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    seq_size = stats['seq_size'][0]
    item_num = stats['item_num'][0]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Tenc(args.hidden_factor, item_num, seq_size,
                 args.dropout_rate, args.diffuser_type, device)
    diff = Consistency(
        args.timesteps, args.w, args.beta_start, args.beta_end, args.beta_sche)
    model.to(device)

    if args.load_model_num:
        path = os.path.join(args.save_model_dir,
                            f'ckpt_{args.load_model_num}.pt')
        model.load_state_dict(torch.load(
            path, map_location=device))
        print(f"Loaded model from {path}")

    if args.eval:
        print('--- Test ---')
        evaluate(model, os.path.join(
            data_directory, 'test_data.df'), diff, device)
        print('------------------')
        sys.exit(0)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.lr, weight_decay=args.l2_decay)

    train_df = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    num_batches = int(train_df.shape[0]/args.batch_size)

    for epoch in range(args.load_model_num, args.load_model_num + args.epoch):
        pbar = tqdm(range(num_batches),
                    desc=f'Epoch {epoch+1}/{args.load_model_num + args.epoch}', unit='batch')
        batch_loss = 0
        N = 0
        for _ in pbar:
            batch = train_df.sample(n=args.batch_size).to_dict()
            seq = torch.LongTensor(
                np.array(list(batch['seq'].values()), dtype=np.int64)).to(device)
            lens = torch.tensor(
                list(batch['len_seq'].values()), dtype=torch.long, device=device)
            tgt = torch.LongTensor(
                np.array(list(batch['next'].values()), dtype=np.int64)).to(device)

            x_start = model.cacu_x(tgt)
            h = model.cacu_h(seq, lens, args.p)

            # consistency training step
            noise = torch.randn_like(x_start)
            t = torch.randint(0, args.timesteps,
                              (args.batch_size,), device=device)
            delta = torch.randint(
                1, args.timesteps, (args.batch_size,), device=device)
            s = torch.clamp(t+delta, max=args.timesteps-1)
            x_t = diff.q_sample(x_start, t, noise)
            x_s = diff.q_sample(x_start, s, noise)
            pred_t = model(x_t, h, t)
            pred_s = model(x_s, h, s)

            # Full-range supervision: anchor both levels to x_start
            loss_sup_t = F.mse_loss(pred_t, x_start)
            loss_sup_s = F.mse_loss(pred_s, x_start)
            # Consistency loss
            loss_cons = F.mse_loss(pred_s, pred_t)
            loss = loss_sup_t + loss_sup_s + loss_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            N += len(batch)
            pbar.set_postfix(loss=batch_loss/N)

        if (epoch + 1) % 1 == 0:
            save_model(model, args.save_model_dir, f'ckpt_{epoch + 1}.pt')

            print('--- Validation ---')
            evaluate(model, os.path.join(
                data_directory, 'val_data.df'), diff, device)
            print('--- Test ---')
            evaluate(model, os.path.join(
                data_directory, 'test_data.df'), diff, device)
            print('------------------')


if __name__ == '__main__':
    args = parse_args()
    main(args)