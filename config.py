import argparse


parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--inChannel', default=100,  type=int)
parser.add_argument('--midChannel', default=100, type=int)
parser.add_argument('--outClust',  default=100, type=int)
parser.add_argument('--in_shape_y',  default=70, type=int)
parser.add_argument('--in_shape_x',  default=70, type=int)
parser.add_argument('--maxIter', default=300, type=int)
parser.add_argument('--minLabels',  default=4, type=int)
parser.add_argument('--lr',  default=0.01, type=float)
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int)
parser.add_argument('--model_dir', metavar='epoch dir', default='model_weights/',
                    help='File path', required=False)

args = parser.parse_args()

