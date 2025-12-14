import json
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.config import parse_structured
from ..utils.typing import *


def _parse_object_list_single(object_list_path: str):
    all_objects = []
    if object_list_path.endswith(".json"):
        with open(object_list_path) as f:
            all_objects = json.loads(f.read())
    else:
        raise NotImplementedError

    return all_objects


def _parse_object_list(object_list_path: Union[str, List[str]]):
    all_objects = []
    if isinstance(object_list_path, str):
        object_list_path = [object_list_path]
    for object_list_path_ in object_list_path:
        all_objects += _parse_object_list_single(object_list_path_)
    return all_objects


def _parse_scene_list_single(scene_list_path: str, root_data_dir: str):
    all_scenes = []
    if scene_list_path.endswith(".json"):
        with open(scene_list_path) as f:
            for p in json.loads(f.read()):
                all_scenes.append(os.path.join(root_data_dir, p))
    elif scene_list_path.endswith(".txt"):
        with open(scene_list_path) as f:
            for p in f.readlines():
                p = p.strip()
                all_scenes.append(os.path.join(root_data_dir, p))
    else:
        raise NotImplementedError

    return all_scenes


def _parse_scene_list(
    scene_list_path: Union[str, List[str]], root_data_dir: Union[str, List[str]]
):
    all_scenes = []
    if isinstance(scene_list_path, str):
        scene_list_path = [scene_list_path]
    if isinstance(root_data_dir, str):
        root_data_dir = [root_data_dir]
    for scene_list_path_, root_data_dir_ in zip(scene_list_path, root_data_dir):
        all_scenes += _parse_scene_list_single(scene_list_path_, root_data_dir_)
    return all_scenes


def random_morphological_transform(
    mask: torch.Tensor, max_kernel_size: int = 5, p_dilation: float = 0.5
) -> torch.Tensor:
    """
    Randomly dilate or erode the mask
    :param mask: [H, W] 0-1 mask
    :param max_kernel_size: Maximum kernel size; controls the scale of the morphological operation
    :param p_dilation: Probability of dilation; if random sample > p_dilation, erosion is performed
    :return: Perturbed mask
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)

    kernel_size = np.random.randint(1, max_kernel_size + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if np.random.rand() < p_dilation:
        mask_np = cv2.dilate(mask_np, kernel, iterations=1)
    else:
        mask_np = cv2.erode(mask_np, kernel, iterations=1)

    return torch.from_numpy(mask_np).float().to(mask.device)


@dataclass
class MultiObjectDataModuleConfig:
    scene_list: Any = ""
    object_list: Any = ""

    # Surface
    surface_root_dir: Any = ""
    surface_suffix: str = "npy"
    num_surface_samples_per_object: int = 20480
    return_scene: bool = False

    max_num_instances: Optional[int] = None
    padding: bool = True
    num_instances_per_batch: Optional[int] = 10

    # Image input
    image_root_dir: Any = ""
    image_prefix: Any = "render"
    image_suffix: str = "webp"
    idmap_prefix: str = "semantic"
    idmap_suffix: str = "png"
    background_color: Union[str, float] = "white"
    image_names: List[str] = field(default_factory=lambda: [])
    height: int = 768
    width: int = 768

    use_scene_image: bool = True
    remove_scene_bg: bool = False

    # Data processing
    skip_small_object: bool = False
    small_image_proportion: float = 0.005  # (16/224)^2

    ## Mask perturbation
    morph_perturb: bool = False
    max_kernel_size: int = 5
    p_dilation: float = 0.5

    return_crop_padded: bool = False
    height_crop_padded: int = 224
    width_crop_padded: int = 224

    # Mix data
    do_mix: bool = False
    do_mix_prob: float = 0.5

    mix_length: int = 80000
    mix_scene_list: str = ""
    mix_image_root_dir: str = ""
    mix_surface_root_dir: str = ""
    mix_surface_suffix: str = "npy"
    mix_image_prefix: str = "render_opaque"
    mix_image_names: List[str] = field(default_factory=lambda: [])
    mix_image_suffix: str = "webp"

    train_indices: Optional[Tuple[Any, Any]] = None
    val_indices: Optional[Tuple[Any, Any]] = None
    test_indices: Optional[Tuple[Any, Any]] = None

    repeat: int = 1

    batch_size: int = 1
    eval_batch_size: int = 1

    num_workers: int = 16


class MultiObjectDataset(Dataset):
    def __init__(self, cfg: Any, split: str = "train") -> None:
        super().__init__()
        assert split in ["train", "val", "test"]
        self.cfg: MultiObjectDataModuleConfig = cfg

        self.all_scenes = _parse_scene_list(
            self.cfg.scene_list, self.cfg.surface_root_dir
        )
        self.all_objects = _parse_object_list(self.cfg.object_list)
        if len(self.all_scenes) != len(self.all_objects):
            raise ValueError(
                f"Number of scenes and objects must be the same, got {len(self.all_scenes)} scenes and {len(self.all_objects)} object lists."
            )

        self.all_images = _parse_scene_list(
            self.cfg.scene_list, self.cfg.image_root_dir
        )

        self.split = split
        self.indices = []
        if self.split == "train" and self.cfg.train_indices is not None:
            self.indices = (self.cfg.train_indices[0], self.cfg.train_indices[1])
        elif self.split == "val" and self.cfg.val_indices is not None:
            self.indices = (self.cfg.val_indices[0], self.cfg.val_indices[1])
        elif self.split == "test" and self.cfg.test_indices is not None:
            self.indices = (self.cfg.test_indices[0], self.cfg.test_indices[1])
        else:
            self.indices = (0, len(self.all_scenes))

        repeat = self.cfg.repeat if self.split == "train" else 1

        self.all_scenes = self.all_scenes[self.indices[0] : self.indices[1]] * repeat
        self.all_objects = self.all_objects[self.indices[0] : self.indices[1]] * repeat
        self.all_images = self.all_images[self.indices[0] : self.indices[1]] * repeat

        if self.cfg.do_mix:
            self.mix_all_scenes = _parse_scene_list(
                self.cfg.mix_scene_list, self.cfg.mix_surface_root_dir
            )[: self.cfg.mix_length]
            self.mix_all_images = _parse_scene_list(
                self.cfg.mix_scene_list, self.cfg.mix_image_root_dir
            )[: self.cfg.mix_length]

    def __len__(self):
        return len(self.all_scenes)

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif bg_color == "random_gray":
            bg_color = random.uniform(0.3, 0.7)
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, list) or isinstance(bg_color, tuple):
            bg_color = np.array(bg_color, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_surface(self, path, num_pc: int = 20480):
        if path.endswith(".npy"):
            data = np.load(path, allow_pickle=True).tolist()
            surface = data["surface_points"]  # Nx3
            normal = data["surface_normals"]  # Nx3
        elif path.endswith(".obj") or path.endswith(".glb"):
            import trimesh

            n_surf_sample = 500000
            scene = trimesh.load(path, process=False, force="scene")
            meshes = []

            for node_name in scene.graph.nodes_geometry:
                geom_name = scene.graph[node_name][1]
                geometry = scene.geometry[geom_name]
                transform = scene.graph[node_name][0]
                if isinstance(geometry, trimesh.Trimesh):
                    geometry.apply_transform(transform)
                    meshes.append(geometry)
            mesh = trimesh.util.concatenate(meshes)
            surface, face_indices = trimesh.sample.sample_surface(
                mesh, n_surf_sample, sample_color=False
            )
            normal = mesh.face_normals[face_indices]
        else:
            raise NotImplementedError(f"Unsupported file format: {path}")

        rng = np.random.default_rng()
        ind = rng.choice(surface.shape[0], num_pc, replace=False)
        surface = torch.FloatTensor(surface[ind])
        normal = torch.FloatTensor(normal[ind])
        surface = torch.cat([surface, normal], dim=-1)

        return surface

    def load_image(
        self,
        path,
        height,
        width,
        background_color,
        rescale: bool = False,
        return_mask: bool = False,
        remove_bg: bool = False,
        idmap_path: Optional[str] = None,
    ):
        image_pil = Image.open(path).resize((width, height))
        image = torch.from_numpy(np.array(image_pil)).float() / 255.0

        if image_pil.mode == "RGBA":
            image_bg = image[:, :, :3] * image[:, :, 3:4] + background_color * (
                1 - image[:, :, 3:4]
            )
            mask = (image[:, :, 3] > 0.5).float()
        elif remove_bg and idmap_path is not None:
            id_map = torch.from_numpy(
                np.array(Image.open(idmap_path).resize((width, height), Image.NEAREST))
            )
            mask = (id_map > 0).float()
            mask_ = mask.unsqueeze(-1).repeat(1, 1, 3)
            image_bg = image * mask_ + background_color * (1 - mask_)
        else:
            image_bg = image
            mask = torch.ones_like(image[:, :, 0]).float()

        if rescale:
            image_bg = image_bg * 2.0 - 1.0
        if return_mask:
            return image_bg, mask
        return image_bg

    def load_parts(
        self,
        rgb_path: str,
        idmap_path: str,
        indexes: List[int],
        height: int,
        width: int,
        background_color: torch.Tensor,
        skip_small_object: bool = False,
        small_image_proportion: float = 0.005,
        morph_perturb: bool = False,  # Whether to apply morphological perturbation
        max_kernel_size: int = 5,
        p_dilation: float = 0.5,
    ):
        rgb_image = self.load_image(rgb_path, height, width, background_color)
        id_map = torch.from_numpy(
            np.array(Image.open(idmap_path).resize((width, height), Image.NEAREST))
        )
        height, width, _ = rgb_image.shape
        rgb_list, mask_list, success_list = [], [], []

        for idx in indexes:
            mask = (id_map == idx).float()

            if (
                skip_small_object
                and mask.sum() <= small_image_proportion * height * width
            ):
                success_list.append(False)
                continue

            if morph_perturb:
                mask = random_morphological_transform(
                    mask, max_kernel_size=max_kernel_size, p_dilation=p_dilation
                )

            mask_3c = mask.unsqueeze(-1).repeat(1, 1, 3)
            part_rgb = rgb_image * mask_3c + background_color * (1 - mask_3c)
            rgb_list.append(part_rgb)
            mask_list.append(mask)
            success_list.append(True)

        return rgb_list, mask_list, success_list

    def crop_and_pad(self, rgbs, masks, height, width, padding_ratio=0.1):
        cropped_rgbs, cropped_masks = [], []

        for rgb, mask in zip(rgbs, masks):
            rgb = rgb.permute(2, 0, 1)

            # crop
            coords = torch.nonzero(mask == 1)
            y_min, x_min = coords.min(dim=0).values
            y_max, x_max = coords.max(dim=0).values

            cropped_rgb = rgb[:, y_min : y_max + 1, x_min : x_max + 1]
            cropped_mask = mask[y_min : y_max + 1, x_min : x_max + 1]

            h, w = cropped_rgb.shape[1:]

            # padding
            padding_size = [0, 0, 0, 0]  # left, right, top, bottom
            if w > h:
                padding_size[2] = padding_size[3] = int((w - h) / 2)
                h = w
            else:
                padding_size[0] = padding_size[1] = int((h - w) / 2)
                w = h

            padding_size = tuple([s + int(w * padding_ratio) for s in padding_size])
            padded_rgb = F.pad(cropped_rgb, padding_size, mode="constant", value=1)
            padded_mask = F.pad(cropped_mask, padding_size, mode="constant", value=0)

            # resize
            padded_rgb = F.interpolate(
                padded_rgb.unsqueeze(0), (height, width), mode="bilinear"
            )[0]
            padded_mask = F.interpolate(
                padded_mask.unsqueeze(0).unsqueeze(0), (height, width), mode="nearest"
            )[0][0]

            cropped_rgbs.append(padded_rgb)
            cropped_masks.append(padded_mask)

        return cropped_rgbs, cropped_masks

    def _getitem_scene(self, index):
        background_color = torch.as_tensor(self.get_bg_color(self.cfg.background_color))

        # Surface
        scene = self.all_scenes[index]
        scene_objects = self.all_objects[index]
        surfaces = []
        for scene_object in scene_objects:
            surface_path = os.path.join(
                scene, f"{scene_object}.{self.cfg.surface_suffix}"
            )
            surface = self.load_surface(
                surface_path, self.cfg.num_surface_samples_per_object
            )
            surfaces.append(surface)
        surfaces = torch.stack(surfaces)  # (num_instances, num_points, 6)

        num_instances = surfaces.shape[0]

        # Image
        image_dir = self.all_images[index]
        image_name = (
            random.choice(self.cfg.image_names)
            if self.split == "train"
            else self.cfg.image_names[0]
        )
        image_prefix = (
            [self.cfg.image_prefix]
            if isinstance(self.cfg.image_prefix, str)
            else self.cfg.image_prefix
        )
        image_prefix = (
            random.choice(image_prefix) if self.split == "train" else image_prefix[0]
        )
        image_path = os.path.join(
            image_dir, f"{image_prefix}_{image_name}.{self.cfg.image_suffix}"
        )
        idmap_path = (
            os.path.join(
                image_dir,
                f"{self.cfg.idmap_prefix}_{image_name}.{self.cfg.idmap_suffix}",
            )
            .replace("_controlnet", "")
            .replace("_inpaint", "")
        )

        # Load image and parts
        rgb_scene = (
            self.load_image(
                image_path,
                height=self.cfg.height,
                width=self.cfg.width,
                background_color=background_color,
                remove_bg=self.cfg.remove_scene_bg,
                idmap_path=idmap_path,
            )
            .unsqueeze(0)
            .repeat(num_instances, 1, 1, 1)
            .permute(0, 3, 1, 2)
        )
        rgbs, masks, success_list = self.load_parts(
            image_path,
            idmap_path,
            list(range(1, num_instances + 1)),
            self.cfg.height,
            self.cfg.width,
            background_color,
            skip_small_object=self.cfg.skip_small_object,
            small_image_proportion=self.cfg.small_image_proportion,
            morph_perturb=self.cfg.morph_perturb,
            max_kernel_size=self.cfg.max_kernel_size,
            p_dilation=self.cfg.p_dilation,
        )
        if len(rgbs) == 0:
            return self._getitem(random.randint(0, self.__len__() - 1))

        rgb = torch.stack(rgbs).permute(0, 3, 1, 2)
        mask = torch.stack(masks).unsqueeze(1)

        # Update `surfaces`, `num_instances`, `rgb_scene` according to `success_list`
        success_list = torch.tensor(success_list, dtype=torch.bool)
        surfaces = surfaces[success_list]
        num_instances = surfaces.shape[0]
        rgb_scene = rgb_scene[success_list]

        if self.cfg.max_num_instances is not None:
            if num_instances > self.cfg.max_num_instances:
                indices = torch.randperm(num_instances)[: self.cfg.max_num_instances]
                surfaces = surfaces[indices]
                rgb = rgb[indices]
                mask = mask[indices]
                rgb_scene = rgb_scene[indices]
                num_instances = self.cfg.max_num_instances

        # Scene id
        scene_id = "-".join(image_dir.split("/")[-2:])

        rv = {
            "id": scene_id,
            "num_instances": num_instances,
            "surface": surfaces,
            "rgb": rgb,
            "mask": mask,
            "rgb_scene": (
                rgb_scene if self.cfg.use_scene_image else torch.zeros_like(rgb_scene)
            ),
        }

        if self.cfg.return_scene:
            surface_scene = surfaces.view(-1, *surfaces.shape[2:])
            rv.update({"surface_scene": surface_scene})

        if self.cfg.return_crop_padded:
            cropped_rgbs, cropped_masks = self.crop_and_pad(
                rgbs, masks, self.cfg.height_crop_padded, self.cfg.width_crop_padded
            )
            cropped_rgb = torch.stack(cropped_rgbs)
            cropped_mask = torch.stack(cropped_masks).unsqueeze(1)

            rv.update(
                {"rgb_crop_padded": cropped_rgb, "mask_crop_padded": cropped_mask}
            )

        if self.cfg.padding:
            keys = [
                "surface",
                "rgb",
                "mask",
                "rgb_scene",
                "rgb_crop_padded",
                "mask_crop_padded",
            ]
            if num_instances < self.cfg.num_instances_per_batch:
                pad = self.cfg.num_instances_per_batch - num_instances
                indices = torch.randint(
                    0, num_instances, (pad,), device=surfaces.device
                )
                updated_dict = {
                    k: torch.cat([v, v[indices]]) if k in keys else v
                    for k, v in rv.items()
                }
            else:
                indices = torch.randperm(num_instances, device=surfaces.device)
                indices = indices[: self.cfg.num_instances_per_batch]
                updated_dict = {
                    k: v[indices] if k in keys else v for k, v in rv.items()
                }
                updated_dict.update({"num_instances": self.cfg.num_instances_per_batch})
            rv = updated_dict

        return rv

    def _getitem_mix(self, index):
        background_color = torch.as_tensor(self.get_bg_color(self.cfg.background_color))

        surfaces, rgbs, masks = [], [], []

        indexes = torch.randint(
            0, len(self.mix_all_scenes), (self.cfg.num_instances_per_batch,)
        )
        for i in indexes:
            scene = self.mix_all_scenes[i]

            # Surface
            surface_path = f"{scene}.{self.cfg.mix_surface_suffix}"
            surface = self.load_surface(
                surface_path, self.cfg.num_surface_samples_per_object
            )
            surfaces.append(surface)

            # Image
            image_dir = self.mix_all_images[i]
            image_name = (
                random.choice(self.cfg.mix_image_names)
                if self.split == "train"
                else self.cfg.mix_image_names[0]
            )
            image_path = os.path.join(
                image_dir,
                f"{self.cfg.mix_image_prefix}_{image_name}.{self.cfg.mix_image_suffix}",
            )

            rgb, mask = self.load_image(
                image_path,
                height=self.cfg.height,
                width=self.cfg.width,
                background_color=background_color,
                return_mask=True,
            )
            rgbs.append(rgb)
            masks.append(mask)

        surfaces = torch.stack(surfaces)  # (num_instances, num_points, 6)
        rgb = torch.stack(rgbs).permute(0, 3, 1, 2)
        mask = torch.stack(masks).unsqueeze(1)

        # Scene ID
        scene_id = self.mix_all_images[indexes[0]].split("/")[-1]

        rv = {
            "id": scene_id,
            "num_instances": 1,
            "surface": surfaces,
            "rgb": rgb,
            "mask": mask,
            "rgb_scene": (
                rgb if self.cfg.use_scene_image else torch.zeros_like(rgb)
            ),  # here single object is the scene
        }

        return rv

    def _getitem(self, index):
        if (
            self.split == "train"
            and self.cfg.do_mix
            and random.random() < self.cfg.do_mix_prob
        ):
            return self._getitem_mix(index)
        else:
            return self._getitem_scene(index)

    def __getitem__(self, index):
        try:
            return self._getitem(index)
        except Exception as e:
            print(f"Error in {self.all_scenes[index]}: {e}")
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        pack = lambda t: t.view(-1, *t.shape[2:])
        for k in batch.keys():
            if k in [
                "surface",
                "rgb",
                "mask",
                "rgb_scene",
                "rgb_crop_padded",
                "mask_crop_padded",
            ]:
                batch[k] = pack(batch[k])

        batch["num_instances_per_batch"] = self.cfg.num_instances_per_batch

        return batch


class MultiObjectDataModule(pl.LightningDataModule):
    cfg: MultiObjectDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(MultiObjectDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = MultiObjectDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MultiObjectDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = MultiObjectDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.val_dataset.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.eval_batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            collate_fn=self.test_dataset.collate,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


if __name__ == "__main__":
    import torchvision
    from omegaconf import OmegaConf

    config_file = "configs/scenediff/training.yaml"
    data_cfg = OmegaConf.load(config_file)["data"]
    cfg: MultiObjectDataModuleConfig = MultiObjectDataModuleConfig(**data_cfg)
    data_module = MultiObjectDataModule(cfg)
    data_module.setup()

    for batch in data_module.test_dataloader():
        print(batch["num_instances"])

        for key in [
            "rgb",
            "mask",
            "rgb_scene",
            # "rgb_crop_padded",
            # "mask_crop_padded",
        ]:
            print(key, batch[key].shape, batch[key].min(), batch[key].max())
            torchvision.utils.save_image(
                batch[key], f"tmp/{key}.png", nrow=4, normalize=True
            )

        for key in ["rgb"]:
            for i in range(batch[key].shape[0]):
                torchvision.utils.save_image(
                    batch[key][i], f"tmp/{key}_{i}.png", normalize=True
                )

        break
