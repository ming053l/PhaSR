import numpy as np
import torchvision.utils as vutils


def get_np_imgrid(array, nrow=3, padding=0, pad_value=0):
    '''
    achieves the same function of torchvision.utils.make_grid for
    numpy array
    '''
    # assume every image has smae size
    n, h, w, c = array.shape
    row_num = n // nrow + (n % nrow != 0)
    gh, gw = row_num*h + padding*(row_num-1), nrow*w + padding*(nrow - 1)
    grid = np.ones((gh, gw, c), dtype=array.dtype) * pad_value
    for i in range(n):
        grow, gcol = i // nrow, i % nrow
        off_y, off_x = grow * (h + padding), gcol * (w + padding)
        grid[off_y : off_y + h, off_x : off_x + w] = array[i]
    return grid


def split_np_imgrid(imgrid, nimg, nrow, padding=0):
    '''
    reverse operation of make_grid.
    args:
        imgrid: HWC image grid
        nimg: number of images in the grid
        nrow: number of columns in image grid
    return:
        images: list, contains splitted images
    '''
    row_num = nimg // nrow + (nimg % nrow != 0)
    gh, gw, _ = imgrid.shape
    h, w = (gh - (row_num-1)*padding)//row_num, (gw - (nrow-1)*padding)//nrow
    images = []
    for gid in range(nimg):
        grow, gcol = gid // nrow, gid % nrow 
        off_i, off_j = grow * (h + padding), gcol * (w + padding)
        images.append(imgrid[off_i:off_i+h, off_j:off_j+w])
    return images


class MDTableConvertor:
    
    def __init__(self, col_num):
        self.col_num = col_num
        
    def _get_table_row(self, items):
        row = ''
        for item in items:
            row += '| {:s} '.format(item)
        row += '|\n'
        return row

    def convert(self, item_list, title=None):
        '''
        args: 
            item_list: a list of items (str or can be converted to str)
            that want to be presented in table.

            title: None, or a list of strings. When set to None, empty title
            row is used and column number is determined by col_num; Otherwise, 
            it will be used as title row, its length will override col_num.

        return: 
            table: markdown table string.
        '''
        table = ''
        if title: # not None or not []  both equal to true
            col_num = len(title)
            table += self._get_table_row(title)
        else:
            col_num=self.col_num
            table += self._get_table_row([' ']*col_num) # empty title row
        table += self._get_table_row(['-'] * col_num) # header spliter
        for i in range(0, len(item_list), col_num):
            table += self._get_table_row(item_list[i:i+col_num])
        return table
    

def visual_dict_to_imgrid(visual_dict, col_num=4, padding=0):
    '''
    args:
        visual_dict: a dictionary of images of the same size
        col_num: number of columns in image grid
        padding: number of padding pixels to seperate images
    '''
    im_names = []
    im_tensors = []
    for name, visual in visual_dict.items():
        im_names.append(name)
        im_tensors.append(visual)
    im_grid = vutils.make_grid(im_tensors,
                               nrow=col_num ,
                               padding=0,
                               pad_value=1.0)
    layout = MDTableConvertor(col_num).convert(im_names)
    
    return im_grid, layout


def count_parameters(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters())
    
    

class WarmupExpLRScheduler(object):
    def __init__(self, lr_start=1e-4, lr_max=4e-4, lr_min=5e-6, rampup_epochs=4, sustain_epochs=0, exp_decay=0.75):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.rampup_epochs = rampup_epochs
        self.sustain_epochs = sustain_epochs
        self.exp_decay = exp_decay
    
    def __call__(self, epoch):
        if epoch < self.rampup_epochs:
            lr = (self.lr_max - self.lr_start) / self.rampup_epochs * epoch + self.lr_start
        elif epoch < self.rampup_epochs + self.sustain_epochs:
            lr = self.lr_max
        else:
            lr = (self.lr_max - self.lr_min) * self.exp_decay**(epoch - self.rampup_epochs - self.sustain_epochs) + self.lr_min
        # print(lr)
        return lr