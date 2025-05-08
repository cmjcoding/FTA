import argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='ICEWS14')
args.add_argument('--batch_size', type=int, default=1000)
args.add_argument('--emb_size', type=int, default=500)
args.add_argument('--max_epochs', type=int, default=5)
args.add_argument('--init', type=float, default=1e-3)
args.add_argument('--lr', type=float, default=1e-1)
args.add_argument('--gpu', type=int, default=0)
args.add_argument('--max_sample', type=int, default=5)
args.add_argument('--seq_len', type=int, default=128)
args.add_argument('--m', type=str, default='train')
args.add_argument('--epoch', type=int, default=0)
args.add_argument('--mi', type=int, default=0)
args.add_argument('--entity_epoch', type=int, default=1)
args.add_argument('--rel_epoch', type=int, default=1)
args.add_argument('--start_epoch', type=int, default=0)
args.add_argument('--test_sample', type=int, default=1)
args.add_argument('--rand_flag', type=bool, default=False)
args.add_argument('--pre_epochs', type=int, default=5)
args.add_argument('--finetune_epochs', type=int, default=5)
args.add_argument('--seed', type=int, default=0)
args.add_argument('--second', type=bool, default=False)
args.add_argument('--fuzzy_threshold', type=float, default=0.3, help='模糊隶属度阈值')
args.add_argument('--type_embed_size', type=int, default=100, help='类型嵌入维度')
args.add_argument('--debug', action='store_true', help='启用调试模式')
args.add_argument('--debug_level',
                 choices=['DEBUG', 'INFO', 'WARNING'],
                 default='INFO',
                 help='调试日志级别')

args = args.parse_args()
