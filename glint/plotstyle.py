import matplotlib.pyplot as plt

def set_my_plot_style():
    """論文用プロット設定を有効化する"""
    
    # plt.style.use('default')
    # plt.style.use(['science', 'nature'])  # Other options: 'grid', 'ieee' (readable when printed in black and white), 'nature' (sans-serif fonts), 'scatter', 'notebook'. Note that this option take longer time to produce plots.
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = 'Times New Roman'
    # For ADC
    plt.rcParams['pdf.fonttype'] = 42  # To embed fonts when saving as pdf
    plt.rcParams['font.family'] = 'Liberation Serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['text.usetex'] = False  # Enable to use some special symbol in Italic font (e.g., theta)
    plt.rcParams['font.size'] = 14  # This parameter control all text size, including title, x-axis and y-axis label, the x-axis and y-axis tick, legend, text, and annotation (default=10). We can change the font size for each element (e.g., title) using another code as followed.
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.direction'] = 'in'  # Change x-axis and y-axis tick direction (in/out/inout)
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 5.0  # Adjust major and minor tick size
    plt.rcParams['xtick.minor.size'] = 3.0  # Default is 3.5 and 2.0
    plt.rcParams['ytick.major.size'] = 5.0
    plt.rcParams['ytick.minor.size'] = 3.0
    plt.rcParams['axes.linewidth'] = 2.  # Adjust the line width of the axes (default is 0.8)
    plt.rcParams['legend.handlelength'] = 0.8  # Adjust handle length
    # plt.rcParams['legend.fancybox'] = False # 丸角
    plt.rcParams['legend.framealpha'] = 0 # 透明度の指定、0で塗りつぶしなし
    # plt.rcParams['legend.edgecolor'] = 'black' # edgeの色を変更
    plt.rcParams['xtick.top'] = True            # 上部に目盛り線を描くかどうか
    plt.rcParams['xtick.bottom'] = True         # 下部に目盛り線を描くかどうか
    plt.rcParams['ytick.left'] = True           # 左部に目盛り線を描くかどうか
    plt.rcParams['ytick.right'] = True          # 右部に目盛り線を描くかどうか
    ##### If need #####
    # ax.xaxis.set_minor_locator(MultipleLocator(.5))  # To show the minor tick
    # ax.yaxis.set_minor_locator(MultipleLocator(.5))
    # plt.savefig('xxx.png', dpi = 300, pad_inches = .1, bbox_inches = 'tight')

    ##### For colorblind-friendly colormap #####
    # One of the safest solutions you can use to deal with color blind is to avoid using red and green together.
    # https://towardsdatascience.com/matplotlib-styles-for-scientific-plotting-d023f74515b4
