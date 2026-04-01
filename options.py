class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--scheduler', type=str, default='reduce_on_plateau',
                            choices=['reduce_on_plateau', 'cosine', 'step'],
                            help='LR scheduler type')
        parser.add_argument('--batch_size', type=int, default=9, help='batch size')
        parser.add_argument('--nepoch', type=int, default=200, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers') 
        parser.add_argument('--eval_workers', type=int, default=2, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default='render_data')
        parser.add_argument('--pretrain_weights', type=str, default=None,help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--arch', type=str, default='phasr', help='archtechture')
        parser.add_argument('--mode', type=str, default='shadow', help='image restoration mode')

        # args for saving
        parser.add_argument('--save_dir', type=str, default='./phasr_log', help='save dir')
        parser.add_argument('--save_images', action='store_true', default=False)
        parser.add_argument('--env', type=str, default='Debug', help='env')
        parser.add_argument('--checkpoint', type=int, default=5, help='checkpoint')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=16, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str, default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')



        # args for training
        parser.add_argument('--debug', action='store_true', default=False, help='debug model')
        parser.add_argument('--eval_now', type=int, default=1, help='After how many trainig step to evalute')
        parser.add_argument('--train_ps', type=int, default=512, help='patch size of training sample')
        parser.add_argument('--resume', action='store_true')

        
        
        parser.add_argument('--train_dir', type=str, default='dataset/train', help='dir of train data')
        parser.add_argument('--val_dir', type=str, default='dataset/valid', help='dir of val data')
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')
        parser.add_argument("--local-rank", type=int, default=0)

        return parser
