import argparse
from collections import namedtuple


parser = argparse.ArgumentParser(description='Define parameters.')

"""Network hyperparameters"""
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--n_step', type=int, default=0)
parser.add_argument('--global_epoch', type=int, default=10)
parser.add_argument('--n_batch', type=int, default=64)
parser.add_argument('--n_img_row', type=int, default=8)
parser.add_argument('--n_img_col', type=int, default=8)
parser.add_argument('--n_img_channels', type=int, default=119)
parser.add_argument('--n_classes', type=int, default=4672)
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--n_resid_units', type=int, default=19)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--dataset', dest='processed_dir', default='./processed_data')
parser.add_argument('--model_path', dest='load_model_path',
                    default='./out/tf')
parser.add_argument('--model_type', dest='model', default='original',
                    help='choose residual block architecture {original,elu,full}')
parser.add_argument('--optimizer', dest='opt', default='sgd')
parser.add_argument('--gtp_policy', dest='gpt_policy', default='mctspolicy',
                    help='choose gtp bot player')  # random,mctspolicy
parser.add_argument('--num_playouts', type=int, dest='num_playouts', default=800,
                    help='The number of MC search per move, the more the better.')
parser.add_argument('--mode', dest='MODE', default='selfplay', help='among selfplay, train and test')
parser.add_argument('--num_maxmoves', type=int, dest='num_maxmoves', default=500,
                    help='The maximum number of moves per game.')
parser.add_argument('--out_dir', dest='out_dir', default='./out/self-play',
                    help='The output directory of generated features and pgns.')

"""Self Play Pipeline"""
parser.add_argument('--N_moves_per_train', dest='N_moves_per_train', type=int, default=4096)
parser.add_argument('--selfplay_games_per_epoch', type=int,
                    dest='selfplay_games_per_epoch', default=1000)

FLAGS = parser.parse_args()

"""ResNet hyperparameters"""
HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer, temperature, global_norm, num_gpu, '
                     'name')

HPS = HParams(batch_size=FLAGS.n_batch,
              num_classes=FLAGS.n_classes,
              min_lrn_rate=0.0002,
              lrn_rate=FLAGS.lr,
              num_residual_units=FLAGS.n_resid_units,
              use_bottleneck=False,
              weight_decay_rate=0.0001,
              relu_leakiness=0,
              optimizer=FLAGS.opt,
              temperature=1.0,
              global_norm=100,
              num_gpu=FLAGS.n_gpu,
              name='01')
