#plot configs
color_dict = {
    'red': '#ff0A47',
    'pink': '#ffc2d1',
    'blue': '#3f88c5',
    'lightblue': '#9fc4e2',
    'yellow': '#fccA46',
    'lightyellow': '#fee5a3',
    'black': '#0d1f2d',
    'gray': '#b3b8bd',
    'purple': '#b47eb4',
    'lightpurple': '#d9bfd9',
    'green': '#36d99a',
    'lightgreen': '#afe9da',
    'virtual': '#227c64'
}

label = {
    'corr': r'Corral',
    'alexp': r'ALEXP',
    'full': r'Naive UCB',
    'oracle': r'Oracle UCB',
    'etc': r'ETC',
    'ets': r'ETS'
}

line_color = {
    'full': color_dict['yellow'],
    'corr': color_dict['blue'],
    'oracle': color_dict['black'],
    'etc': color_dict['red'],
    'ets': color_dict['purple'],
    'alexp': color_dict['green']
}


shade_color = {
    'corr': color_dict['lightblue'],
    'full':color_dict['lightyellow'],
    'oracle': color_dict['gray'],
    'etc': color_dict['pink'],
    'ets': color_dict['lightpurple'],
    'alexp': color_dict['lightgreen']
}


line_styles = {
    'full': ':',
    'corr': '--',
    'oracle': '-',
    'etc': '-.'
}

linestyles = [':', '--', '-.', '-', '-']