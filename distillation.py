import numpy as np
import pandas as pd
import random
import sys
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import logging
from tenc import Tenc
from diffusion import Student, Teacher
from utility import evaluate, save_model, setup_seed

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
    parser.add_argument('--w', type=float, default=2.0, help='guidance weight')
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
                        default=0, help='Load model checkpoint')
    parser.add_argument('--eval', action='store_true',
                        default=False, help='Evaluate the model')
    parser.add_argument('--teacher_ckpt', type=str,
                        default=None, help='Path to teacher model for distillation')
    parser.add_argument('--infer_steps', type=int,
                        default=4, help='student infer steps')

    return parser.parse_args()


# —— main training + distillation loop —— #
def main(args):
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_directory = f'./data/yc'

    stats = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
    seq_size, item_num = stats['seq_size'][0], stats['item_num'][0]

    diff_student = Student(args.timesteps, args.beta_start,
                           args.beta_end, args.beta_sche)
    diff_teacher = Teacher(args.timesteps, args.beta_start,
                           args.beta_end, args.beta_sche, args.w)

    # — load teacher — #
    teacher = Tenc(args.hidden_factor, item_num, seq_size,
                   args.dropout_rate, args.diffuser_type, device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # — student — #
    student = Tenc(args.hidden_factor, item_num, seq_size,
                   args.dropout_rate, args.diffuser_type, device).to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            student.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            student.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            student.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            student.parameters(), lr=args.lr, weight_decay=args.l2_decay)

    if args.load_model_num:
        path = os.path.join(args.save_model_dir,
                            f'ckpt_{args.load_model_num}.pt')
        student.load_state_dict(torch.load(
            path, map_location=device))
        print(f"Loaded model from {path}")

    if args.eval:
        print('--- Test ---')
        evaluate(student, os.path.join(data_directory, 'test_data.df'),
                 diff_student, device, args.infer_steps)
        print('------------------')
        sys.exit(0)

    df_train = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    nb = len(df_train) // args.batch_size

    for epoch in range(args.epoch):
        pbar = tqdm(range(nb),
                    desc=f'Epoch {epoch+1}/{args.epoch}', unit='batch')
        batch_loss = 0
        N = 0
        for _ in pbar:
            batch = df_train.sample(n=args.batch_size).to_dict()
            seq = torch.LongTensor(
                np.array(list(batch['seq'].values()), dtype=np.int64)).to(device)
            lens = torch.tensor(
                list(batch['len_seq'].values()), dtype=torch.long, device=device)
            tgt = torch.LongTensor(
                np.array(list(batch['next'].values()), dtype=np.int64)).to(device)

            # 1) sample two levels t < s
            t_val = random.randint(0, args.timesteps-2)
            s_val = random.randint(t_val+1, args.timesteps-1)
            # t_steps = torch.full((args.batch_size,), t_val, device=device, dtype=torch.long)
            s_steps = torch.full((args.batch_size,), s_val,
                                 device=device, dtype=torch.long)

            # 2) get embeddings
            x0 = student.cacu_x(tgt)
            h = student.cacu_h(seq, lens, args.p)

            # 3) noise to x_s
            x_s = diff_teacher.q_sample(x0, s_steps)

            # 4) teacher rollback x_s -> x_t
            x_t = x_s
            for n in reversed(range(t_val+1, s_val+1)):
                ti = torch.full((args.batch_size,), n,
                                device=device, dtype=torch.long)
                x_t = diff_teacher.p_sample(
                    teacher.forward, teacher.forward_uncon, x_t, h, ti, n)

            # 5) student predict x_t from x_s at level s
            pred = student.forward(x_s, h, s_steps)

            # 6) distillation loss
            loss = F.mse_loss(pred, x_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()
            N += len(batch)

            pbar.set_postfix(loss=batch_loss/N)

        if (epoch + 1) % 1 == 0:
            save_model(student, args.save_model_dir, f'ckpt_{epoch + 1}.pt')
            print('----- VALIDATION -----')
            evaluate(student, os.path.join(data_directory, 'val_data.df'),
                     diff_student, device, args.infer_steps)
            print('----- TEST -----')
            evaluate(student, os.path.join(data_directory,
                     'test_data.df'), diff_student, device, args.infer_steps)


if __name__ == '__main__':
    args = parse_args()
    main(args)
