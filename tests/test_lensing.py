import unittest

import numpy as np
from scipy.ndimage import map_coordinates

from glint.lensing import map_source_to_image_cube


def reference_map_coordinates_loop(beta_x, beta_y, source_cube, pixscale, order):
    """旧実装（channelごとのmap_coordinates）をそのまま再現した参照実装。
    """
    nch, ny_src, nx_src = source_cube.shape
    beta_x_pix = beta_x / pixscale + nx_src / 2.0
    beta_y_pix = beta_y / pixscale + ny_src / 2.0
    coords = np.stack([beta_y_pix, beta_x_pix])
    out = np.empty((nch, *beta_x.shape), dtype=source_cube.dtype)
    for channel in range(nch):
        out[channel] = map_coordinates(
            source_cube[channel],
            coords,
            order=order,
            mode="constant",
            cval=0.0,
            prefilter=(order > 1),
        )
    return out


class MapSourceToImageCubeTest(unittest.TestCase):
    """order=1のgather高速パスが旧map_coordinates実装と同値であることの回帰テスト。
    """

    def setUp(self):
        # setUpは各テストメソッドの直前に毎回呼ばれ、入力(fixture)を新しく作る。
        rng = np.random.default_rng(20260817)
        self.nch, self.ny_src, self.nx_src = 5, 40, 30
        cube = rng.normal(size=(self.nch, self.ny_src, self.nx_src))
        # source cubeの端はゼロ余白（map_source_to_image_cubeのdocstringの前提）
        cube[:, :2, :] = 0.0
        cube[:, -2:, :] = 0.0
        cube[:, :, :2] = 0.0
        cube[:, :, -2:] = 0.0
        self.cube32 = cube.astype(np.float32)
        self.cube64 = cube.astype(np.float64)
        self.pixscale = 0.01
        # source gridの内側・外側の両方に落ちる座標を含める。
        # あえて正方形ではない画像座標を使うことで、axis順序のバグもチェック
        ny_img, nx_img = 25, 35
        self.beta_x = rng.uniform(-0.25, 0.25, size=(ny_img, nx_img))
        self.beta_y = rng.uniform(-0.25, 0.25, size=(ny_img, nx_img))

    def test_order1_matches_map_coordinates_float32(self):
        """本番と同じfloat32 cubeでのテスト

        float32の丸め誤差（cube値~1に対して相対~1e-7）に余裕を持たせたatol=5e-6で比較する。
        rtol=0にしているのは、値がほぼ0のpixelで相対誤差が発散して偽陽性になるのを避けるため。
        """
        fast = map_source_to_image_cube(
            self.beta_x, self.beta_y, self.cube32, self.pixscale, order=1,
        )
        reference = reference_map_coordinates_loop(
            self.beta_x, self.beta_y, self.cube32, self.pixscale, order=1,
        )
        np.testing.assert_allclose(fast, reference, rtol=0.0, atol=5e-6)

    def test_order1_matches_map_coordinates_float64(self):
        """float64 cubeでの同値性。atol=1e-12とfloat32版よりずっと厳しい。

        これが通ることは、高速パスの内部でうっかりfloat32へ落としていない（入力のdtypeで重み計算している）ことの検出にもなる。
        もし内部にfloat32キャストが混入すると、誤差が~1e-7程度になりatol=1e-12で落ちる。
        """
        fast = map_source_to_image_cube(
            self.beta_x, self.beta_y, self.cube64, self.pixscale, order=1,
        )
        reference = reference_map_coordinates_loop(
            self.beta_x, self.beta_y, self.cube64, self.pixscale, order=1,
        )
        np.testing.assert_allclose(fast, reference, rtol=0.0, atol=1e-12)

    def test_order1_outside_source_grid_is_zero(self):
        """βがsource gridの遥か外（±10 arcsec、gridは±0.5 arcsec）なら出力が厳密に0になること。

        高速パス内のinsideマスクを狙い撃ちするテスト。
        実装は範囲外indexをnp.clipで安全な値に丸めてからgatherし、最後にinsideマスクで0にするという2段構えなので、「clipはしたがマスクを掛け忘れた」というバグがあるとgridの端の値が漏れて出力が非0になり、ここで検出される。
        こちらは丸め誤差の余地がない（0×何か=0）のでassert_array_equalで厳密比較。
        """
        beta_far_x = np.full((4, 4), 10.0)
        beta_far_y = np.full((4, 4), -10.0)
        fast = map_source_to_image_cube(
            beta_far_x, beta_far_y, self.cube32, self.pixscale, order=1,
        )
        np.testing.assert_array_equal(fast, 0.0)

    def test_order2_falls_back_to_map_coordinates(self):
        """order>=2が従来のmap_coordinates（spline）経路のまま変わっていないこと。

        分岐を追加したときに壊しやすいのは「触っていないはずの側」なので、fallback経路の挙動も固定しておく。
        """
        out = map_source_to_image_cube(
            self.beta_x, self.beta_y, self.cube32, self.pixscale, order=2,
        )
        reference = reference_map_coordinates_loop(
            self.beta_x, self.beta_y, self.cube32, self.pixscale, order=2,
        )
        np.testing.assert_allclose(out, reference, rtol=0.0, atol=5e-6)


if __name__ == "__main__":
    unittest.main()
