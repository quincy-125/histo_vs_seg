"""Copyright 2024 The University of Pittsburgh. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
"""

from typing import List, Set, Dict, Union, Tuple, Optional

import os
import dataclasses
import PIL
import skimage
import openslide
import numpy as np


@dataclasses.dataclass
class WSIPatchExtract:
    """Extract image patches for either the H&E or mIF WSI."""

    def __init__(
        self,
        thumbnail_width: int = 500,
        patch_level: int = 0,
        max_patch_size: int = 1024,
        tissue_area_threshold: float = 0.9,
    ) -> None:
        super(WSIPatchExtract).__init__()
        self.thumbnail_width = thumbnail_width
        self.patch_level = patch_level
        self.max_patch_size = max_patch_size
        self.tissue_area_threshold = tissue_area_threshold

    def _get_wsi_thumbnail(
        self,
        wsi_dir: str,
    ) -> Union[openslide.OpenSlide, List[int], PIL.Image.Image]:
        """Create the thumbnail image for WSI.

        Parameters
        ----------
        wsi_dir : str
            directory of input WSI.

        Returns
        -------
        Union[openslide.OpenSlide, list[int], PIL.Image.Image]
            return the openslide WSI object, top_level with patch width,
            height, and the ratio for the target image width.
        """
        wsi = openslide.OpenSlide(filename=wsi_dir)

        # Get the ratio for the target image width.
        divisor = int(wsi.level_dimensions[0][0] / self.thumbnail_width)
        # Get the height and width of the thumbnail using the ratio.
        patch_size_x = int(wsi.level_dimensions[0][0] / divisor)
        patch_size_y = int(wsi.level_dimensions[0][1] / divisor)
        top_level = [patch_size_x, patch_size_y, divisor]
        # Extract the thumbnail.
        thumbnail = wsi.get_thumbnail(size=(patch_size_x, patch_size_y))
        return wsi, top_level, thumbnail

    def _get_otsu_binary_img(self, thumbnail: PIL.Image.Image) -> np.ndarray:
        """Create binary mask image for tissue detection using OTSU method.

        Parameters
        ----------
        thumbnail : PIL.Image.Image
            The thumbnail image of WSI.

        Returns
        -------
        np.ndarray
            binary mask image in numpy array.
        """
        # Convert to grey scale image.
        gs_thumbnail = np.array(thumbnail.convert("L"))
        # Get the otsu threshold value.
        thresh = skimage.filters.threshold_otsu(image=gs_thumbnail)
        # Convert to binary mask.
        binary_img = gs_thumbnail < thresh
        binary_img = binary_img.astype(int)
        return binary_img

    def _locate_tissue_regions(self, binary_img: np.ndarray) -> Set[Tuple[int]]:
        """Locate the x- and y- coords from binary image mask with tissue pixel values.

        Parameters
        ----------
        binary_img : np.ndarray
            binary mask image array.

        Returns
        -------
        set[Tuple[int]]
            set of x- and y- coords for tissue regions.
        """
        idx = np.sum(binary_img)
        idx = np.where(binary_img == 1)
        tissue_region_index = []
        for i in range(0, len(idx[0]), 1):
            x = idx[1][i]
            y = idx[0][i]
            tissue_region_index.append((x, y))

        return set(tissue_region_index)

    def _get_patch_coords(
        self,
        wsi: openslide.OpenSlide,
        divisor: int,
        tissue_region_index: Set[Tuple[int]],
    ) -> Set[Tuple[int]]:
        """Get the x- and y- coords for patches that needs to be extracted.

        Parameters
        ----------
        wsi : openslide.OpenSlide
            openslide WSI object.
        divisor : int
            ratio for the target image width.
        tissue_region_index : set[Tuple[int]]
            set of x- and y- coords for tissue regions.

        Returns
        -------
        Set[Tuple[int]]
            set of tuple of x- and y- coords for patches that needs to be extracted.
        """
        assert (
            self.patch_level <= len(wsi.level_dimensions) - 1
        ), f"level {str(self.patch_level)} exceeds {str(len(wsi.level_dimensions)-1)}"

        patch_start_xy_coords = []

        # creating sub patches
        # Iterating through x coordinate
        wsi_level_dims = wsi.level_dimensions[self.patch_level]
        wsi_level_downsamples = wsi.level_downsamples[self.patch_level]

        start_x = 0
        while start_x + self.max_patch_size < wsi_level_dims[0]:
            # Iterating through y coordinate
            start_y = 0
            while start_y + self.max_patch_size < wsi_level_dims[1]:
                current_x = int(start_x * wsi_level_downsamples / divisor)
                current_y = int(start_y * wsi_level_downsamples / divisor)
                current_x_stop = int(
                    ((start_x + self.max_patch_size) * wsi_level_downsamples) / divisor
                )
                current_y_stop = int(
                    ((start_y + self.max_patch_size) * wsi_level_downsamples) / divisor
                )

                tissue_pixels = sum(
                    (i, j) in tissue_region_index
                    for i in range(current_x, current_x_stop + 1)
                    for j in range(current_y, current_y_stop + 1)
                )

                if (
                    tissue_pixels
                    / (
                        (current_y_stop + 1 - current_y)
                        * (current_x_stop + 1 - current_x)
                    )
                ) > self.tissue_area_threshold:
                    patch_start_xy_coords.append((start_x, start_y))
                start_y += self.max_patch_size
            start_x += self.max_patch_size
        return set(patch_start_xy_coords)

    def _extract_patches(
        self,
        wsi_dir: str,
        wsi: openslide.OpenSlide,
        patch_start_xy_coords: Set[Tuple[int]],
    ) -> Union[str, Dict[str, np.ndarray]]:
        """Extract image patches from WSI.

        Parameters
        ----------
        wsi_dir : str
            directory of input WSI.
        wsi : openslide.OpenSlide
            openslide WSI object.
        patch_start_xy_coords : Set[Tuple[int]]
            set of tuple of x- and y- coords for patches that needs to be extracted.

        Returns
        -------
        Union[str, dict[str, np.ndarray]]
            return the unique identifier for the de-identified WSI, and the
            dictionary with key be patch name, value be the patch object in np.ndarray.
        """
        wsi_uuid = wsi_dir.split("/")[-1].split(".")[0]
        patch_names = []
        patch_objs = []

        for coords in patch_start_xy_coords:
            start_x_coord = int(coords[0] * wsi.level_downsamples[self.patch_level])
            start_y_coord = int(coords[-1] * wsi.level_downsamples[self.patch_level])
            patch_obj = wsi.read_region(
                location=(start_x_coord, start_y_coord),
                level=self.patch_level,
                size=(self.max_patch_size, self.max_patch_size),
            )
            assert patch_obj.size == (
                self.max_patch_size,
                self.max_patch_size,
            ), f"the actual patch size {patch_obj.size} is not as expected {self.max_patch_size}"
            patch_name = f"{wsi_uuid}_level_{self.patch_level}_x_{start_x_coord}_y_{start_y_coord}"

            patch_names.append(patch_name)
            patch_objs.append(np.asarray(patch_obj))
        return wsi_uuid, dict(zip(patch_names, patch_objs))

    def _save_patches(
        self, wsi_uuid: str, patch_dicts: Dict[str, np.ndarray], patch_dir: str
    ) -> None:
        """Save extracted image patches.

        Parameters
        ----------
        wsi_uuid: str
            unique identifier for the de-identified WSI.
        patch_dicts : dict[str, np.ndarray]
            dictionary with key be patch name, value be the patch object in np.ndarray.
        patch_dir : str
            directory of output image patches extracted from WSI.
        """
        os.makedirs(f"{patch_dir}/{wsi_uuid}", exist_ok=True)
        for name, patch in patch_dicts.items():
            np.save(f"{patch_dir}/{wsi_uuid}/{name}.npy", patch)

    def forward(
        self, input_dir: str, output_dir: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Main function to execute the patch extraction from WSIPatchExtract class.

        Returns
        -------
        Optional[dict[str, np.ndarray]]
            dictionary with key be patch name, value be the patch object in np.ndarray.
        """
        wsi, top_level, thumbnail = self._get_wsi_thumbnail(input_dir)
        binary_img = self._get_otsu_binary_img(thumbnail)
        tissue_region_index = self._locate_tissue_regions(binary_img)
        patch_start_xy_coords = self._get_patch_coords(
            wsi, top_level[-1], tissue_region_index
        )
        wsi_uuid, patch_dicts = self._extract_patches(
            input_dir, wsi, patch_start_xy_coords
        )
        if output_dir is None:
            return patch_dicts
        self._save_patches(wsi_uuid, patch_dicts, output_dir)
        return None


@dataclasses.dataclass
class ROIPatchExtract(WSIPatchExtract):
    """Extract image patches for the manual selected ROIs from either the H&E or mIF WSI."""

    def __init__(self, thumbnail_width: int = 500, max_patch_size: int = 1024) -> None:
        super(WSIPatchExtract).__init__()
        self.thumbnail_width = thumbnail_width
        self.max_patch_size = max_patch_size

    def _get_roi_thumbnail(
        self,
        roi_dir: str,
    ) -> Union[PIL.Image.Image, PIL.Image.Image]:
        """Create the thumbnail image for manual selected ROI from WSI.

        Parameters
        ----------
        roi_dir : str
            directory of input manual selected ROI from WSI.

        Returns
        -------
        Union[PIL.Image.Image, PIL.Image.Image]
            return the openslide WSI object, and the ratio for the target image width.
        """
        roi = PIL.Image.open(roi_dir)
        # Extract the thumbnail.
        thumbnail = roi.copy()
        try:
            thumbnail.thumbnail((self.thumbnail_width, self.thumbnail_width))
        except ValueError:
            thumbnail = (
                f"thumbnail image is not supported when input image mode is {roi.mode}"
            )
        return roi, thumbnail

    def _get_roi_patch_coords(self, roi: PIL.Image.Image) -> Set[Tuple[int]]:
        """Get the x- and y- coords for patches that needs to be extracted.

        Parameters
        ----------
        roi : PIL.Image.Image
            PIL.Image.Image object of manual selected ROI from WSI.

        Returns
        -------
        Set[Tuple[int]]
            set of tuple of x- and y- coords for patches that needs to be extracted.
        """
        patch_start_x_coords = set(range(0, roi.size[0], self.max_patch_size))
        patch_start_y_coords = set(
            y for y in range(0, roi.size[-1], self.max_patch_size)
        )

        patch_xy_coords = [
            (x, y, x + self.max_patch_size, y + self.max_patch_size)
            for x in patch_start_x_coords
            for y in patch_start_y_coords
        ]
        return set(patch_xy_coords)

    def _extract_roi_patches(
        self,
        roi_dir: str,
        roi: PIL.Image.Image,
        patch_xy_coords: Set[Tuple[int]],
    ) -> Union[str, Dict[str, np.ndarray]]:
        """Extract image patches from WSI.

        Parameters
        ----------
        roi_dir : str
             directory of input manual selected ROI from WSI.
        roi : PIL.Image.Image
            PIL.Image.Image object of manual selected ROI from WSI.
        patch_xy_coords : Set[Tuple[int]]
            set of tuple of x- coords for patches that needs to be extracted.

        Returns
        -------
        Union[str, dict[str, np.ndarray]]
            return the unique identifier for the de-identified WSI, and the
            dictionary with key be patch name, value be the patch object in np.ndarray.
        """
        wsi_uuid = roi_dir.split("/")[-1].split(".")[0]
        patch_names = []
        patch_objs = []

        for coords in patch_xy_coords:
            patch_obj = roi.crop(coords)
            assert patch_obj.size == (
                self.max_patch_size,
                self.max_patch_size,
            ), f"the actual patch size {patch_obj.size} is not as expected {self.max_patch_size}"
            patch_name = f"{wsi_uuid}_roi__x_{coords[0]}_y_{coords[1]}"
            patch_names.append(patch_name)
            patch_objs.append(np.asarray(patch_obj))
        return wsi_uuid, dict(zip(patch_names, patch_objs))

    def forward(
        self, input_dir: str, output_dir: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Main function to execute the patch extraction from WSIPatchExtract class.

        Returns
        -------
        Optional[dict[str, np.ndarray]]
            dictionary with key be patch name, value be the patch object in np.ndarray.
        """
        roi, _ = self._get_roi_thumbnail(input_dir)
        patch_xy_coords = self._get_roi_patch_coords(roi)

        wsi_uuid, patch_dicts = self._extract_roi_patches(
            input_dir, roi, patch_xy_coords
        )
        if output_dir is None:
            return patch_dicts
        self._save_patches(wsi_uuid, patch_dicts, output_dir)
        return None
