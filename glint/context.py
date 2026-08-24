from __future__ import annotations
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Optional, Sequence, Tuple, List
import numpy as np
from scipy.fft import fftn, next_fast_len, rfftn
import finufft

# -----------------------------
# Utility
# -----------------------------
def _as_float32_c(a: np.ndarray) -> np.ndarray:
    """float32のC連続配列にそろえ、実行時の暗黙コピーを避ける。"""
    return np.asarray(a, dtype=np.float32, order="C")


def _as_bool(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=bool, order="C")


def _as_int(value: int, name: str) -> int:
    """floatを暗黙に切り捨てず、整数だけを受け付ける。"""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _as_positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return value



# -----------------------------
# Image Contexts
# -----------------------------
''' ForwardModel class in forward.py に置き換え
@dataclass(frozen=True, slots=True)
class ImageContext:
    """
    forward_model_image のための固定変数をまとめたクラス

    サンプリング中に変わらないものだけ入れる：
      - grid (xx_img, yy_img, xx_src, yy_src)
      - pixelsize
      - channel info（vchan, raw correlator channel spacing）
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

    # dirty beam kernel (3D) [nch, pix, pix]：image-domain で畳み込みするなら
    beam: Optional[np.ndarray] = None
    # cached FFT of beam
    beam_fft: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    beam_rfft: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    fft_shape: Optional[Tuple[int, int]] = field(default=None, init=False)

    # ------------- 3D only -------------
    # spectral axis
    vchan_kms: Optional[np.ndarray] = None  # shape (nchan,)
    raw_channel_spacing_kms: Optional[float] = None  # online averaging前のraw channel spacing [km/s]

    # radial grid
    radius_arcsec: Optional[np.ndarray] = None  # shape (nr,)

    def __post_init__(self):
        object.__setattr__(self, "xx_img", _as_float32_c(self.xx_img))
        object.__setattr__(self, "yy_img", _as_float32_c(self.yy_img))
        object.__setattr__(self, "xx_src", _as_float32_c(self.xx_src))
        object.__setattr__(self, "yy_src", _as_float32_c(self.yy_src))

        for name in ("xx_img", "yy_img", "xx_src", "yy_src"):
            grid = getattr(self, name)
            if grid.ndim != 2 or 0 in grid.shape:
                raise ValueError(f"{name} must be a non-empty 2D array.")
            if not np.all(np.isfinite(grid)):
                raise ValueError(f"{name} contains NaN or inf.")
        if self.xx_img.shape != self.yy_img.shape:
            raise ValueError("xx_img and yy_img must have same shape.")
        if self.xx_src.shape != self.yy_src.shape:
            raise ValueError("xx_src and yy_src must have same shape.")
        if (
            not np.isfinite(self.pixsize_img)
            or not np.isfinite(self.pixsize_src)
            or self.pixsize_img <= 0
            or self.pixsize_src <= 0
        ):
            raise ValueError("pixsize_img/pixsize_src must be finite and > 0.")

        # optional arrays
        if self.vchan_kms is not None:
            object.__setattr__(self, "vchan_kms", _as_float32_c(self.vchan_kms))
            if self.vchan_kms.ndim != 1 or self.vchan_kms.size == 0:
                raise ValueError("vchan_kms must be a non-empty 1D array.")
            if not np.all(np.isfinite(self.vchan_kms)):
                raise ValueError("vchan_kms contains NaN or inf.")

        if self.radius_arcsec is not None:
            object.__setattr__(self, "radius_arcsec", _as_float32_c(self.radius_arcsec))
            if self.radius_arcsec.ndim != 1 or self.radius_arcsec.size == 0:
                raise ValueError("radius_arcsec must be a non-empty 1D array.")
            if not np.all(np.isfinite(self.radius_arcsec)):
                raise ValueError("radius_arcsec contains NaN or inf.")

        if self.raw_channel_spacing_kms is not None and (
            not np.isfinite(self.raw_channel_spacing_kms)
            or self.raw_channel_spacing_kms < 0
        ):
            raise ValueError("raw_channel_spacing_kms must be finite and >= 0.")
        for name in ("x0_src", "y0_src", "x0_l", "y0_l"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")

        if self.beam is not None:
            beam = np.asarray(self.beam, dtype=np.float32, order="C")
            object.__setattr__(self, "beam", beam)

            # image size
            ny, nx = self.xx_img.shape

            # beam size
            if beam.ndim == 2:
                ky, kx = beam.shape
            elif beam.ndim == 3:
                nbeam, ky, kx = beam.shape
                if self.vchan_kms is not None and nbeam != self.nchan:
                    raise ValueError(
                        "a 3D beam must have one plane per velocity channel."
                    )
            else:
                raise ValueError("beam must be 2D or 3D array.")
            if ky == 0 or kx == 0:
                raise ValueError("beam must not have empty spatial dimensions.")
            if not np.all(np.isfinite(beam)):
                raise ValueError("beam contains NaN or inf.")

            # FFTのコストは O(L^2 log L) で効くので注意
            # L = next_fast_len(n + k - 1) とするとcrop外のところまで厳密に計算することになり無駄
            # cropで使うのは index [s, s + n) だけ（s = k//2 はn/2中心規約でのkernel原点。CASA PSFのpeak位置と一致し、奇数kernelでは(k-1)//2と同値）。 <-- 地味に重要、注意しないと1pixズレる
            # 簡単な例 : 1D, n=6, k=4, peakがc=2=k//2にあるkernel:

            # I = [0, 0, 1, 0, 0, 0]          源は j0=2
            # B = [.1, .5, 1, .5]             peak は c=2
            # full conv C = [0, 0, .1, .5, 1, .5, 0, 0, 0]   peakは m = j0+c = 4

            # scipy 'same' (s=(k-1)//2=1): C[1:7] = [0, .1, .5, 1, .5, 0]  → peakが i=3 No +1ずれ
            # s = k//2 = 2:                C[2:8] = [.1, .5, 1, .5, 0, 0]  → peakが i=2 OK


            # 巡回畳み込みのalias汚染は先頭 [0, n+k-1-L) に限られる。
            # 以下の２条件を満たせばいい：
            # 1. alias汚染がcrop内に入らないこと --> n + k - 1 - L <= s すなわち L >= n + k - 1 - s
            # 2. cropの右端が配列内に収まること --> L >= s + n
            # これでもcrop内はfull linear convolution (L >= n+k-1) と厳密に一致する。
            Ly = next_fast_len(max(ny + ky - 1 - ky // 2, ky // 2 + ny)) # next_fast_lenは切り上げなので大丈夫
            Lx = next_fast_len(max(nx + kx - 1 - kx // 2, kx // 2 + nx))

            beam_fft = fftn(beam, s=(Ly, Lx), axes=(-2, -1)) # 0-paddingしている
            beam_rfft = rfftn(beam, s=(Ly, Lx), axes=(-2, -1))

            object.__setattr__(self, "beam_fft", beam_fft)
            object.__setattr__(self, "beam_rfft", beam_rfft)
            object.__setattr__(self, "fft_shape", (Ly, Lx))

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
'''



# -----------------------------
# UV Contexts
# -----------------------------
@dataclass(frozen=True, slots=True)
class UVContext:
    """
    visibility-domain forward model の固定変数。

    FFTの固定部分だけ入れる：
      - primary beam (pb)
      - FINUFFT plans / slices / Ntot
      - flag
      - (optional) flatten済みのdata/sigma（``build_uv_observation``が設定）
    """
    # primary beam on image grid: (ny_img, nx_img) or (nchan, ny_img, nx_img)
    pb: np.ndarray

    # FINUFFT plans per channel: list[Plan|None]
    plans: Sequence[Optional[object]]

    # Flatten layout for concatenated vis: list[slice|None] (per channel)
    slices: Sequence[Optional[slice]]

    # number of channels
    nchan: int

    # total number of unflagged visibilities across all channels
    Ntot: int

    # (optional) keep flag per channel for debug/plotting
    flag: Optional[np.ndarray] = None  # shape (nchan, Nrow)

    # FINUFFT transform type used by plans
    nufft_type: int = 3

    # (optional) flatten済みの観測visibilityと1 sigma。``slices``と同じ並びで、
    # ``build_uv_observation``が構築時に整合を保証する（演算子だけの用途ではNone）。
    data: Optional[np.ndarray] = None   # (Ntot,) complex
    sigma: Optional[np.ndarray] = None  # (Ntot,) per-component std [Jy]

    def __post_init__(self):
        nchan = _as_int(self.nchan, "nchan")
        ntotal = _as_int(self.Ntot, "Ntot")
        nufft_type = _as_int(self.nufft_type, "nufft_type")
        object.__setattr__(self, "pb", _as_float32_c(self.pb))
        object.__setattr__(self, "plans", tuple(self.plans))
        object.__setattr__(self, "slices", tuple(self.slices))
        object.__setattr__(self, "nchan", nchan)
        object.__setattr__(self, "Ntot", ntotal)
        object.__setattr__(self, "nufft_type", nufft_type)
        if self.flag is not None:
            object.__setattr__(self, "flag", _as_bool(self.flag))

        if self.nchan <= 0:
            raise ValueError("nchan must be > 0.")
        if len(self.plans) != self.nchan:
            raise ValueError("len(plans) must equal nchan.")
        if len(self.slices) != self.nchan:
            raise ValueError("len(slices) must equal nchan.")
        if self.Ntot < 0:
            raise ValueError("Ntot must be >= 0.")
        if self.nufft_type not in (2, 3):
            raise ValueError("nufft_type must be 2 or 3.")
        for name in ("data", "sigma"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != (ntotal,):
                raise ValueError(f"{name} must have shape (Ntot,) = ({ntotal},).")

        if self.pb.ndim == 2:
            pass
        elif self.pb.ndim == 3 and self.pb.shape[0] == self.nchan:
            pass
        else:
            raise ValueError(
                "pb must have shape (ny_img, nx_img) or "
                "(nchan, ny_img, nx_img)."
            )
        if 0 in self.pb.shape[-2:]:
            raise ValueError("pb must not have empty spatial dimensions.")
        if not np.all(np.isfinite(self.pb)):
            raise ValueError("pb contains NaN or inf.")

        if self.flag is not None:
            if self.flag.ndim != 2 or self.flag.shape[0] != self.nchan:
                raise ValueError("flag must have shape (nchan, nrow).")

        expected_start = 0
        for channel, (plan, output_slice) in enumerate(
            zip(self.plans, self.slices, strict=True)
        ):
            if (plan is None) != (output_slice is None):
                raise ValueError(
                    f"channel {channel}: plan and slice must both be None "
                    "or both be non-None."
                )
            plan_type = getattr(plan, "type", None)
            if plan_type is not None and plan_type != self.nufft_type:
                raise ValueError(
                    f"channel {channel}: plan type {plan_type} does not match "
                    f"nufft_type={self.nufft_type}."
                )

            nvalid = None
            if self.flag is not None:
                nvalid = int(np.count_nonzero(~self.flag[channel]))

            if output_slice is None:
                if nvalid not in (None, 0):
                    raise ValueError(
                        f"channel {channel}: {nvalid} unflagged values but no slice."
                    )
                continue

            if not isinstance(output_slice, slice):
                raise ValueError(
                    f"channel {channel}: slices entries must be slice or None."
                )
            if output_slice.step not in (None, 1):
                raise ValueError(f"channel {channel}: slice step must be 1.")
            if (
                isinstance(output_slice.start, bool)
                or not isinstance(output_slice.start, Integral)
                or isinstance(output_slice.stop, bool)
                or not isinstance(output_slice.stop, Integral)
            ):
                raise ValueError(
                    f"channel {channel}: slice start/stop must be integers."
                )
            if output_slice.start != expected_start:
                raise ValueError(
                    f"channel {channel}: slice starts at {output_slice.start}, "
                    f"expected {expected_start}."
                )
            if output_slice.stop <= expected_start:
                raise ValueError(
                    f"channel {channel}: an active slice must not be empty."
                )

            slice_length = output_slice.stop - expected_start
            if nvalid is not None and slice_length != nvalid:
                raise ValueError(
                    f"channel {channel}: slice length {slice_length} does not "
                    f"match {nvalid} unflagged values."
                )
            expected_start = output_slice.stop

        if expected_start != self.Ntot:
            raise ValueError(
                f"slices contain {expected_start} visibilities but Ntot={self.Ntot}."
            )

    @property
    def image_shape(self) -> Tuple[int, int]:
        return tuple(int(value) for value in self.pb.shape[-2:])

    def primary_beam(self, channel: int) -> np.ndarray:
        return self.pb if self.pb.ndim == 2 else self.pb[channel]


# -----------------------------
# Builders
# -----------------------------
def build_uv_layout(
        u: np.ndarray, 
        v: np.ndarray, 
        flag: np.ndarray
) -> Tuple[List[Optional[Tuple[np.ndarray, np.ndarray]]], 
           List[Optional[slice]], 
           int]:
    """
    計算の途中で不変なk_list / slices / Ntot を作る
    """
    u = np.asarray(u)
    v = np.asarray(v)
    flag = _as_bool(flag)
    if u.ndim != 2 or v.ndim != 2 or flag.ndim != 2:
        raise ValueError("u, v, and flag must be 2D arrays (nchan, nrow).")
    if not (u.shape == v.shape == flag.shape):
        raise ValueError("u, v, and flag must have the same shape.")
    if u.shape[0] == 0:
        raise ValueError("u, v, and flag must contain at least one channel.")
    valid = ~flag
    if not np.all(np.isfinite(u[valid])) or not np.all(np.isfinite(v[valid])):
        raise ValueError("unflagged u/v coordinates contain NaN or inf.")

    nchan = u.shape[0]

    k_list: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    slices: List[Optional[slice]] = []

    start = 0

    for i in range(nchan):
        m = ~flag[i]
        ni = int(m.sum())
        if ni == 0:
            k_list.append(None)
            slices.append(None)
            continue
        # 必ずinterferometer.py と同じ規約でFINUFFTの座標系に変換する。
        kx = -(2.0 * np.pi * u[i, m]).astype(np.float32, order="C")
        ky =  (2.0 * np.pi * v[i, m]).astype(np.float32, order="C")
        k_list.append((kx, ky))
        slices.append(slice(start, start + ni))
        start += ni

    return k_list, slices, start


def build_finufft_plans(
    k_list: Sequence[Optional[Tuple[np.ndarray, np.ndarray]]],
    l_rad: Optional[np.ndarray] = None,
    m_rad: Optional[np.ndarray] = None,
    eps: float = 1e-6,
    *,
    nufft_type: int = 3,
    image_shape: Optional[Tuple[int, int]] = None,
    pixsize_arcsec: Optional[float] = None,
) -> List[Optional[object]]:
    """チャンネルごとに再利用可能なtype-2/type-3 planを作る。"""
    nufft_type = _as_int(nufft_type, "nufft_type")
    if nufft_type not in (2, 3):
        raise ValueError("nufft_type must be 2 or 3.")
    eps = _as_positive_float(eps, "eps")

    if nufft_type == 2:
        if image_shape is None or len(image_shape) != 2:
            raise ValueError("image_shape must be (ny, nx) for type-2 plans.")
        ny, nx = (_as_int(value, "image_shape") for value in image_shape)
        if ny <= 0 or nx <= 0:
            raise ValueError("image_shape must be (ny, nx) for type-2 plans.")
        if pixsize_arcsec is None:
            raise ValueError("pixsize_arcsec is required for type-2 plans.")
        pixsize_arcsec = _as_positive_float(pixsize_arcsec, "pixsize_arcsec")
        if nx % 2 or ny % 2:
            raise ValueError(
                "type-2 plans require even image dimensions for the "
                "make_grid_arcsec center convention."
            )
        pixsize_rad = pixsize_arcsec * np.deg2rad(1.0 / 3600.0)
    else:
        if l_rad is None or m_rad is None:
            raise ValueError("l_rad and m_rad are required for type-3 plans.")
        l_rad = _as_float32_c(l_rad)
        m_rad = _as_float32_c(m_rad)
        if l_rad.size == 0 or l_rad.shape != m_rad.shape:
            raise ValueError("l_rad and m_rad must have the same non-empty shape.")
        if not np.all(np.isfinite(l_rad)) or not np.all(np.isfinite(m_rad)):
            raise ValueError("l_rad/m_rad contain NaN or inf.")
        l_rad = l_rad.ravel()
        m_rad = m_rad.ravel()

    nchan = len(k_list)
    plans = [None] * nchan

    for i in range(nchan):
        if k_list[i] is None:
            continue
        kx, ky = k_list[i]
        kx = _as_float32_c(kx)
        ky = _as_float32_c(ky)
        if kx.ndim != 1 or ky.ndim != 1 or kx.size == 0 or kx.shape != ky.shape:
            raise ValueError(
                f"channel {i}: kx and ky must be same-length non-empty 1D arrays."
            )
        if not np.all(np.isfinite(kx)) or not np.all(np.isfinite(ky)):
            raise ValueError(f"channel {i}: kx/ky contain NaN or inf.")
        plan = finufft.Plan(
            nufft_type=nufft_type,
            n_modes_or_dim=(ny, nx) if nufft_type == 2 else 2,
            isign=+1,
            eps=eps,
            dtype=np.complex64,
        )
        if nufft_type == 2:
            # image[y, x]に合わせてFINUFFTの座標順を(y, x)にする。
            plan.setpts(
                x=_as_float32_c(ky * pixsize_rad),
                y=_as_float32_c(kx * pixsize_rad),
            )
        else:
            plan.setpts(x=l_rad, y=m_rad, s=kx, t=ky)
        plans[i] = plan

    return plans


def flatten_selected(array: np.ndarray, flag: np.ndarray, slices) -> np.ndarray:
    # (nchan, nrow)配列のnon-flag要素を抜き出して、slicesと同じ並びの1次元にする
    array = np.asarray(array)
    output = np.empty(
        sum(int(np.count_nonzero(~row)) for row in flag), dtype=array.dtype,
    )
    for channel, output_slice in enumerate(slices):
        if output_slice is not None:
            output[output_slice] = array[channel, ~flag[channel]]
    return output


def build_uv_observation(
    *,
    u_lam: np.ndarray,          # (nchan, nrow) [wavelengths]
    v_lam: np.ndarray,          # (nchan, nrow) [wavelengths]
    data: np.ndarray,           # (nchan, nrow) complex visibilities
    sigma: np.ndarray,          # (nchan, nrow) per-component std [Jy]
    flag: np.ndarray,           # (nchan, nrow) True=使わない
    pb: np.ndarray,             # image grid上のprimary beam (ny,nx) or (nchan,ny,nx)
    image_shape: Tuple[int, int],
    pixsize_arcsec: float,
    nufft_type: int = 2,
    eps: float = 1e-6,
    xx_arcsec: Optional[np.ndarray] = None,  # type-3のときのみ必要
    yy_arcsec: Optional[np.ndarray] = None,
) -> UVContext:
    """uv fitの固定部分（FINUFFT演算子＋flatten済みdata/sigma）を持つUVContextを構築する

    channel順・flag処理・FINUFFT planの整合はここで一括して保証される
    """
    k_list, slices, ntotal = build_uv_layout(u_lam, v_lam, flag)
    arcsec2rad = np.deg2rad(1.0 / 3600.0)
    plans = build_finufft_plans(
        k_list,
        l_rad=None if xx_arcsec is None else xx_arcsec * arcsec2rad,
        m_rad=None if yy_arcsec is None else yy_arcsec * arcsec2rad,
        nufft_type=nufft_type,
        image_shape=image_shape,
        pixsize_arcsec=float(pixsize_arcsec),
        eps=float(eps),
    )
    return UVContext(
        pb=pb, plans=plans, slices=slices,
        nchan=int(np.asarray(data).shape[0]), Ntot=ntotal, flag=flag,
        nufft_type=nufft_type,
        data=flatten_selected(data, flag, slices),
        sigma=flatten_selected(sigma, flag, slices),
    )
