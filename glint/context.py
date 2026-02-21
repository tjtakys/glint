from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List
import numpy as np

# -----------------------------
# Utility
# -----------------------------
def _as_float32_c(a: np.ndarray) -> np.ndarray:
    """Ensure float32 + C-order (FINUFFTやNumPy演算の無駄コピーを減らす)."""
    return np.asarray(a, dtype=np.float32, order="C")


def _as_bool(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=bool, order="C")



# -----------------------------
# Contexts
# -----------------------------
@dataclass(frozen=True, slots=True)
class ImageContext:
    """
    forward_model_image のための固定変数をまとめたクラス

    「毎回変わらないもの」だけ入れる：
      - grid (xx_img, yy_img, xx_src, yy_src)
      - pixelsize
      - channel info（vchan, spec_res）
      - lens center（x0_l, y0_l）
      - beam（image fitting で使うなら）
    """
    # ------------- Common for both of 2D and 3D -------------
    # grids [arcsec]
    xx_img: np.ndarray
    yy_img: np.ndarray
    xx_src: np.ndarray
    yy_src: np.ndarray

    # pixel scales [arcsec/pix]
    pixsize_img: float
    pixsize_src: float

    # source-plane origin [arcsec] (map_source_to_image_cube に必要)
    x0_src: float = 0.0
    y0_src: float = 0.0

    # lens center [arcsec]（HST Gaussian fitとかで固定）
    x0_l: float = 0.0
    y0_l: float = 0.0

    # clean beam kernel (2D) [pix, pix]：image-domain で畳み込みするなら
    beam: np.ndarray = None

    # ------------- 3D only -------------
    # spectral axis
    vchan_kms: Optional[np.ndarray] = None  # shape (nchan,)
    spec_res_sgm_kms: Optional[float] = None  # channel width [km/s]

    # radial grid
    radius_arcsec: Optional[np.ndarray] = None  # shape (nr,)

    def __post_init__(self):
        object.__setattr__(self, "xx_img", _as_float32_c(self.xx_img))
        object.__setattr__(self, "yy_img", _as_float32_c(self.yy_img))
        object.__setattr__(self, "xx_src", _as_float32_c(self.xx_src))
        object.__setattr__(self, "yy_src", _as_float32_c(self.yy_src))

        # optinal arrays
        if self.vchan_kms is not None:
            object.__setattr__(self, "vchan_kms", _as_float32_c(self.vchan_kms))

        if self.radius_arcsec is not None:
            object.__setattr__(self, "radius_arcsec", _as_float32_c(self.radius_arcsec))

        if self.beam is not None:
            object.__setattr__(self, "beam", np.asarray(self.beam, dtype=np.float32, order="C"))

        # sanity checks
        if self.xx_img.shape != self.yy_img.shape:
            raise ValueError("xx_img and yy_img must have same shape.")
        if self.xx_src.shape != self.yy_src.shape:
            raise ValueError("xx_src and yy_src must have same shape.")
        if self.pixsize_img <= 0 or self.pixsize_src <= 0:
            raise ValueError("pixsize_img/pixsize_src must be > 0.")
        if self.spec_res_sgm_kms <= 0:
            raise ValueError("spec_res_sgm_kms must be > 0.")

    @property
    def nchan(self) -> int:
        if self.vchan_kms is None:
            raise ValueError("vchan_kms is not set.")
        return int(self.vchan_kms.size)

    @property
    def img_shape(self) -> Tuple[int, int]:
        return (int(self.xx_img.shape[0]), int(self.xx_img.shape[1]))

    @property
    def src_shape(self) -> Tuple[int, int]:
        return (int(self.xx_src.shape[0]), int(self.xx_src.shape[1]))


@dataclass(frozen=True, slots=True)
class UVContext:
    """
    visibility-domain forward model の固定変数。

    観測の固定部分だけ入れる：
      - primary beam (pb)
      - FINUFFT plans / slices / Ntot
      - flag
    """
    # primary beam on image grid (shape: ny_img, nx_img)
    pb: np.ndarray

    # FINUFFT plans per channel: list[Plan|None]
    plans: Sequence[object]

    # Flatten layout for concatenated vis: list[slice|None] (per channel)
    slices: Sequence[Optional[slice]]

    # number of channels
    nchan: int

    # total number of unflagged visibilities across all channels
    Ntot: int

    # (optional) keep flag per channel for debug/plotting
    flag: Optional[np.ndarray] = None  # shape (nchan, Nrow)

    def __post_init__(self):
        object.__setattr__(self, "pb", _as_float32_c(self.pb))
        if self.flag is not None:
            object.__setattr__(self, "flag", _as_bool(self.flag))

        if len(self.plans) != self.nchan:
            raise ValueError("len(plans) must equal nchan.")
        if len(self.slices) != self.nchan:
            raise ValueError("len(slices) must equal nchan.")

        if self.Ntot < 0:
            raise ValueError("Ntot must be >= 0.")
        if self.pb.ndim != 2:
            raise ValueError("pb must be 2D (ny_img, nx_img).")


# -----------------------------
# Builders
# -----------------------------
def build_uv_layout(u: np.ndarray, v: np.ndarray, flag: np.ndarray) -> Tuple[List[Optional[Tuple[np.ndarray, np.ndarray]]], List[Optional[slice]], int]:
    """
    uv_list / slices / Ntot
    を作るだけの関数（Planの作成は別関数に分離）。

    Parameters
    ----------
    u, v : arrays shaped (nchan, nrow) [in wavelengths]
    flag : bool array shaped (nchan, nrow) True=flagged

    Returns
    -------
    uv_list : list[(kx,ky)|None], where kx=2πu, ky=2πv (float32)
    slices  : list[slice|None]
    Ntot    : total unflagged count
    """
    nchan = u.shape[0]
    uv_list: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    slices: List[Optional[slice]] = []
    start = 0

    for i in range(nchan):
        m = ~flag[i]
        ni = int(m.sum())
        if ni == 0:
            uv_list.append(None)
            slices.append(None)
            continue
        kx = (2.0 * np.pi * u[i, m]).astype(np.float32, order="C")
        ky = (2.0 * np.pi * v[i, m]).astype(np.float32, order="C")
        uv_list.append((kx, ky))
        slices.append(slice(start, start + ni))
        start += ni

    return uv_list, slices, start