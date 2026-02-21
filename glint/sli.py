"""
Warren & Dye (2003) の一番簡単な Semi-Linear Inversion (SLI) の実装
"""

import numpy as np

def build_forward_matrix(lens_model, source_grid):
    """
    
    """
    # ここでは、単純な例として、レンズモデルが単位行列であると仮定します。
    # 実際の実装では、レンズモデルに基づいて前方行列を構築する必要があります。
    return np.eye(len(source_grid))