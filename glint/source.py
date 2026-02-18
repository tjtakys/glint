import numpy as np
from scipy.special import erf, gammaincinv


###################
# 2D model
###################
def sersic2d(x, y, I, x0, y0, ellip, pa, r_eff, n):
    """
    Sérsic 2D surface brightness (astropy.modeling.models.Sersic2D but c=0 fixed)

    Parameters
    ----------
    x, y : array-like
        座標（ブロードキャスト可）
    I : float
        I(r_eff) Jy/arcsec^2 **MUST BE BRIGHTNESS (関数形自体は単位依存ないけど)**
    x0, y0 : float
        中心座標
    ellip : float in [0, 1)
        楕円率 = 1 - b/a   （軸比 q=b/a）
    pa : float [rad]
        PA（+x から反時計回り）
    r_eff : float (>0)
        half-light radius (major axis基準、minor axisは (1-ellip)*r_eff)
    n : float (>0.1)
        Sérsic index
    
    Returns
    -------
    I : ndarray
        x, y とブロードキャストした輝度
    """

    # b_n
    # 厳密な定義式はこっち
    # bn = gammaincinv(2.0*n, 0.5)

    # 近似式 (Ciotti & Bertin 1999; n>0で良好)
    bn = 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - 2194697/(30690717750*n**4)

    # 回転
    ct, st = np.cos(pa), np.sin(pa)
    dx = x - x0
    dy = y - y0
    x_maj =  ct*dx + st*dy
    x_min = -st*dx + ct*dy
    b_semi = (1.0 - ellip) * r_eff

    # 楕円半径
    z = np.sqrt((x_maj / r_eff)**2 + (x_min / b_semi)**2)
    return I * np.exp(-bn * (z**(1.0/n) - 1.0))


def total_flux(map, pixsize_arcsec):
    return map.sum() * (pixsize_arcsec**2)  # Jy/arcsec^2 -> Jy



###################
# 3D model
###################
# q_int（固有軸比）オプションを追加して再保存
import importlib.util, sys, numpy as np
from scipy.special import erf
from typing import Optional, Union, Callable, Tuple
import inspect

import numpy as np
from scipy.special import erf

def make_rotating_disk_cube(
    XX, YY,                 # (ny, nx)   事前作成の像面座標[arcsec]（右=+x, 上=+y）
    x0, y0,                 # ()         中心座標
    vchan_kms,              # (nchan,)   速度チャネル中心[km/s]（観測キューブの軸に合わせる） *** 昇順じゃないと負になるので注意 ***
    inc_deg, pa_deg,        # ()         傾斜角i[deg], PA[deg]（+xから反時計回り）
    radius,                 # (Nr,)      半径サンプル[arcsec]
    sb_profile,             # (Nr,)      面輝度 S(R)（任意スケール; チャネル和がS）
    vrot_profile,           # (Nr,)      回転曲線 Vc(R) [km/s]
    sigma_profile,          # (Nr,)      速度分散 σ_gas(R) [km/s]
    systemic_kms=0.0,       # ()         系統速度 V_sys [km/s]
    rmax_as=None,           # () or None 有効半径上限[arcsec]
    spec_res_sgm_kms=0.0,       # ()     装置LSFのσ[km/s]（FWHMなら/2.355して渡す）
):
    """
    Parameters
    ----------
    XX, YY : 2D array [arcsec]
        画像座標（右=+x, 上=+y）
    x0, y0 : float [arcsec]
        ディスクの中心座標
    vchan_kms : 1D array
        速度チャネル中心 [km/s]
    inc_deg : float
        傾斜角 (0=face-on)
    pa_deg : float
        銀河長軸のPA [deg]、+xから反時計回り
    radius : 1D array
        半径サンプル [arcsec]
    sb_profile, vrot_profile, sigma_profile : 1D array
        半径に対応する S(R), Vc(R), σ(R)
    systemic_kms : float
        系統速度 [km/s]
    rmax_as : float or None
        有効半径上限 [arcsec] (Noneならradiusの最大値)
    spec_res_sgm_kms : float
        装置LSFのσ [km/s] (FWHMなら/2.355して渡す) Roman-Olivia+2023 Appendix 1 も参照。
    """
    # 結果格納用の空キューブ
    ny, nx = XX.shape
    nchan = len(vchan_kms)
    cube = np.zeros((nchan, ny, nx), dtype=np.float32)
    
    # 座標変換
    phi = np.deg2rad(pa_deg)
    cosp, sinp = np.cos(phi), np.sin(phi)
    Xp =  (XX - x0) * cosp + (YY - y0) * sinp # (ny, nx)
    Yp = -(XX - x0) * sinp + (YY - y0) * cosp # (ny, nx)

    inc = np.deg2rad(inc_deg)
    cosi = np.clip(np.cos(inc), 1e-4, 1.0)
    sini = np.sin(inc)

    R = np.sqrt(Xp**2 + (Yp/cosi)**2).astype(np.float32)  # (ny, nx)
    theta = np.arctan2(Yp/cosi, Xp).astype(np.float32)    # (ny, nx)

    if rmax_as is None:
        rmax_as = float(np.max(radius)) 
    mask = (R <= rmax_as)                # (ny, nx) 

    # 計算するピクセルだけ選ぶ
    Xp = Xp[mask]      # (nvalid,)
    Yp = Yp[mask]      # (nvalid,)
    R = R[mask]        # (nvalid,)
    theta = theta[mask]# (nvalid,)

    # プロファイル補間
    S   = np.interp(R, radius, sb_profile).astype(np.float32)  # (nvalid,)
    Vc  = np.interp(R, radius, vrot_profile).astype(np.float32)  # (nvalid,)
    Sig = np.interp(R, radius, sigma_profile).astype(np.float32)  # (nvalid,)

    # LSFを分散に合成（ガウシアン仮定）
    Sig_eff = np.sqrt(Sig**2 + spec_res_sgm_kms**2).astype(np.float32)  # (nvalid,)
    # Sig_eff = np.maximum(Sig_eff, 10)

    # LOS velocity
    Vlos = (Vc * sini * np.cos(theta) + systemic_kms).astype(np.float32) # (nvalid,)
    # チャネル境界（vchan_kms は中心値）
    v = np.asarray(vchan_kms, dtype=np.float32)  # (nchan,)
    edges = np.empty(v.size+1, dtype=np.float32) # (nchan+1,)
    edges[1:-1] = 0.5*(v[:-1]+v[1:])
    edges[0]    = v[0]  - 0.5*(v[1]-v[0])
    edges[-1]   = v[-1] + 0.5*(v[-1]-v[-2])

    # broadcastingで一気に計算
    mu  = Vlos[None, :]    # (1, nvalid)
    sig = Sig_eff[None, :] # (1, nvalid)

    # ガウスCDF差（チャネル端で積分）で配分
    z_hi = (edges[1:, None] - mu)/sig
    z_lo = (edges[:-1, None] - mu)/sig
    w = 0.5 *(erf(z_hi / np.sqrt(2.0)) - erf(z_lo / np.sqrt(2.0)))  # (nchan, nvalid)
    flux = S[None, :] * w  # (nchan, nvalid)

    # キューブに戻す
    cube[:, mask] = flux

    return cube


def Vrot_Courteau1997(r, v_c, r_turn, gamma, beta=1.0):
    """
    Courteau (1997) の回転曲線モデル
    元の式は x=r_turn/rの形で、r=0で問答無用で発散するので、逆数に変数変換した。
    この時、β>1ではr->0でV->∞に発散  β=1でr->0でV->v_c  β<1でr->0でV->0に収束
    したがって、2通りの方法がある。
    1. β=1に固定して、r=0でも計算できるようにする
    2. βを自由にして、最小radiusはdrの半分に設定するなどして、r>0でのみ計算するようにする。これだと、グリッドサイズとの兼ね合いで中心が抜ける恐れがある。
    とりあえず、1.を採用する。
    Parameters
    ----------
    r : array-like
        半径 [arcsec]
    v_c : float
        遠方での回転速度 [km/s]
    r_turn : float
        ターンオーバー半径 [arcsec]
    gamma : float
        sharpness（大きいほど急峻）
    beta : float
        曲線形状パラメータ（負なら減速、正なら上昇、0なら平坦）
    
    Returns
    -------
    Vc : array-like
        回転速度 [km/s]
    """
    y = r / r_turn
    V_r = v_c * (1 + y)**beta * (1 + y**gamma)**(-1/gamma) * y**(1-beta)
    return V_r

def Vrot_arctan(r, v_c, r_turn):
    """
    arctan 回転曲線モデル (Courteau 1997 の特殊ケース)
    Parameters
    ----------
    r : array-like
        半径 [arcsec]
    v_c : float
        遠方での回転速度 [km/s]
    r_turn : float
        ターンオーバー半径 [arcsec]
    
    Returns
    -------
    Vc : array-like
        回転速度 [km/s]
    """
    V_r = (2.0/np.pi) * v_c * np.arctan(r / r_turn)
    return V_r