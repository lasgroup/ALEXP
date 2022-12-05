import torch
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#plot configs
color_dict = {
    'night': '#419d78',
    'eggplant': '#d33f49',
    'bruise': '#e17a81',
    'blood': '#f9a620',
    'bliss': '#8ccfb4',
    'lilly': '#fbc774',
    'morning': '#4f7992'
}

regret_label = {
    'select': r'$R_t^{\mathrm{MS}}-Model Select$',
    'full': r'$R_t(k_{\mathrm{ALL}}) - All Features$',
    'oracle': r'$R_t^* - Oracle$'
}

shade_color = {
    'select': color_dict['bliss'],
    'full':color_dict['bruise'],
    'oracle': color_dict['lilly']
}

line_color = {
    'full': color_dict['eggplant'],
    'select': color_dict['night'],
    'oracle': color_dict['blood'],
}


linestyles = [':', '--', '-.', '-', '-']