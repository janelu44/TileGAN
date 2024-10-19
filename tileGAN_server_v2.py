import glob
import os
import pickle
import socket
from bisect import bisect
import time
from pathlib import Path
from multiprocessing.managers import BaseManager
from joblib import load
from sklearn.cluster import KMeans
from PIL import Image

import h5py
import hnswlib
import numpy as np
import tensorflow as tf

MINIBATCH_SIZE = 4
DEPTHS = [512, 512, 512, 512, 512, 256, 128, 64, 32]


def parse_dataset(dataset_path):
    files = glob.glob(os.path.join(dataset_path, '*network.pkl'))
    if len(files) > 0:
        network_file = files[0]
    else:
        print(f'No network file found at {dataset_path}')
        return None

    files = glob.glob(os.path.join(dataset_path, '*descriptors.hdf5'))
    if len(files) > 0:
        descriptors_file = files[0]
    else:
        print(f'No descriptor file found at {dataset_path}')
        return None

    files = glob.glob(os.path.join(dataset_path, '*clusters.hdf5'))
    clusters_file = f'{dataset_path}_clusters.hdf5' if len(files) == 0 else files[0]

    files = glob.glob(os.path.join(dataset_path, '*kmeans.joblib'))
    kmeans_file = f'{dataset_path}_kmeans.joblib' if len(files) == 0 else files[0]

    files = glob.glob(os.path.join(dataset_path, '*ann.bin'))
    ann_file = f'{dataset_path}_ann.bin' if len(files) == 0 else files[0]

    return network_file, descriptors_file, clusters_file, kmeans_file, ann_file


class TileGanManager:
    def __init__(self, ):
        self.available_datasets = []
        self.selected_dataset = None

        self.find_available_datasets()
        if len(self.available_datasets) == 0:
            print(f'Found no datasets')
            return

        print(f'Found datasets: {self.available_datasets}')

        # Network
        self.network_file = None

        # Descriptors
        self.descriptor_lookup = None
        self.latent_lookup = None
        self.cluster_lookup = None
        self.t_size = 0

        # Clusters
        self.latent_images = None
        self.average_images = None
        self.num_clusters = 0
        self.latent_clusters = []
        self.latent_cdfs = []

        # ANN
        self.ann_numbers = None

        # K-Means
        self.kmeans = None

        self.load_dataset(self.available_datasets[0])

        # Params
        self.merge_level = 2
        self.latent_depth = int(DEPTHS[self.merge_level - 1])
        self.latent_size = 2
        self.height = 0
        self.width = 0
        self.output_resolution = 0

        # Outputs & intermediates
        self.dominant_cluster_colors = []
        self.cluster_grid = None
        self.cluster_samples = None

        self.latents = None

        self.intermediate_latents = None
        self.intermediate_latent_grid = None

        self.output = None

        self.guidance_image = None

        self._load_instance()

    def _save_instance(self):
        with open('instance.pkl', 'wb') as f:
            pickle.dump((
                self.merge_level,
                self.latent_depth,
                self.latent_size,
                self.height,
                self.width,
                self.output_resolution,
                self.dominant_cluster_colors,
                self.cluster_grid,
                self.cluster_samples,
                self.latents,
                self.intermediate_latents,
                self.intermediate_latent_grid,
                self.output,
                self.guidance_image,
                self.selected_dataset
            ), f)
        print('Saved instance.')

    def _load_instance(self):
        if not os.path.exists('instance.pkl'):
            return

        with open('instance.pkl', 'rb') as f:
            instance = pickle.load(f)
            self.merge_level = instance[0]
            self.latent_depth = instance[1]
            self.latent_size = instance[2]
            self.height = instance[3]
            self.width = instance[4]
            self.output_resolution = instance[5]
            self.dominant_cluster_colors = instance[6]
            self.cluster_grid = instance[7]
            self.cluster_samples = instance[8]
            self.latents = instance[9]
            self.intermediate_latents = instance[10]
            self.intermediate_latent_grid = instance[11]
            self.output = instance[12]
            self.guidance_image = instance[13]
            self.selected_dataset = instance[14]
        print('Loaded instance.')

    def find_available_datasets(self, dataset_dir='data'):
        self.available_datasets = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
        return self.available_datasets, self.selected_dataset

    def load_dataset(self, dataset_name, dataset_dir='data'):
        print(f'Loading dataset: {dataset_name}')
        parsed = parse_dataset(os.path.join(dataset_dir, dataset_name))
        if parsed is None:
            print(f'> Failed to load dataset.')
            return

        self.selected_dataset = dataset_name

        network_file, descriptors_file, clusters_file, kmeans_file, ann_file = parsed

        self.dominant_cluster_colors = []
        self.network_file = network_file

        # Load descriptors
        file = h5py.File(descriptors_file, 'r')
        self.descriptor_lookup = file['descriptors'].value
        self.latent_lookup = file['latents'].value
        self.cluster_lookup = file['clusters'].value
        self.t_size = int(np.sqrt(len(self.descriptor_lookup[0]) / 3))
        print('\t> Loaded descriptors.')

        # Load clusters
        file = h5py.File(clusters_file, 'r')
        self.latent_images = file['images'].value
        self.average_images = file['averages'].value
        self.num_clusters = len(self.latent_images)
        for c in range(self.num_clusters):
            self.latent_clusters.append(file[f'{c}'])
            self.latent_cdfs.append(file[f'{c}_cdf'])
        print(f'\t> Loaded {self.num_clusters} clusters.')

        # Load ANN
        dimensions = len(self.descriptor_lookup[0])
        self.ann_numbers = hnswlib.Index(space='l2', dim=dimensions)

        if os.path.isfile(ann_file):
            self.ann_numbers.load_index(ann_file)
            print('\t> Loaded existing ANN index.')
        else:
            print(f'\t> Creating new ANN index for {len(self.descriptor_lookup)} descriptors of dimension {dimensions}...')
            ef_construction = 2000  # reasonable range: 100-2000
            ef_search = 2000  # reasonable range: 100-2000 (higher = better recall but longer retrieval time)
            m = 100  # reasonable range: 5-100 (higher = more accuracy, longer retrieval time)
            self.ann_numbers.init_index(max_elements=len(self.descriptor_lookup), ef_construction=ef_construction, M=m)
            self.ann_numbers.add_items(self.descriptor_lookup, np.arange(len(self.descriptor_lookup)))
            self.ann_numbers.set_ef(ef_search)
            self.ann_numbers.save_index(ann_file)
            print('\t> Created ANN index.')

        # Load K-Means
        if os.path.isfile(kmeans_file):
            self.kmeans = load(kmeans_file)
            print('\t> Loaded existing KMeans.')
        else:
            print(f'\t> Creating {self.num_clusters}-means for {len(self.descriptor_lookup)} descriptors.')
            self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.descriptor_lookup)
            print('\t> Created KMeans.')

        print(f'\tDone!\n')

    def randomize_latents(self, height, width, repeat=False):
        self.height = height
        self.width = width

        self.cluster_grid = np.zeros((self.height * self.latent_size, self.width * self.latent_size), dtype=np.uint8)

        if repeat:
            r = np.random.randint(len(self.latent_lookup))
            latent = self.latent_lookup[r]
            cluster = self.cluster_lookup[r]
            self.latents = np.tile(latent, (self.height, self.width))
            self.cluster_grid = np.tile(cluster, (self.height * self.latent_size, self.width * self.latent_size))
        else:
            rs = np.random.randint(len(self.latent_lookup), size=self.height * self.width)
            self.latents = np.asarray([self.latent_lookup[r] for r in rs])
            clusters = np.asarray([self.cluster_lookup[r] for r in rs])

            ls = self.latent_size
            for x in np.arange(self.width):
                for y in np.arange(self.height):
                    self.cluster_grid[y * ls: (y + 1) * ls, x * ls: (x + 1) * ls] = clusters[y * self.width + x]

    def calculate_intermediate_latents(self, latents):
        if len(latents.shape) == 1:
            latents = np.expand_dims(latents, axis=0)
        if len(latents.shape) > 2:
            latents = np.squeeze(latents)

        with tf.Session(graph=tf.Graph()) as session:
            with open(self.network_file, 'rb') as file:
                network = pickle.load(file)
                size = network.output_shape[-1]
                channels = network.output_shape[1]

                self.output_resolution = int(np.log2(size))
                kwargs = {
                    'in_res': 2,
                    'out_res': self.merge_level,
                    'latent_depth': 512,
                    'label_size': 0,
                    'num_channels': channels,
                    'resolution': size
                }
                gsa = network.clone_and_update("GsA", kwargs=kwargs, func='networks.G_new')

                intermediate_latents, _ = gsa.run_with_session(
                    session,
                    latents,
                    in_res=2,
                    out_res=self.merge_level,
                    latent_depth=512,
                    minibatch_size=MINIBATCH_SIZE,
                    num_gpus=1,
                    out_dtype=np.float32
                )

        session.close()
        return intermediate_latents

    def get_output_from_intermediate_latents(self, intermediate_latents):
        w = self.width
        h = self.height
        ls = self.latent_size
        pad = self.get_left_padding()

        self.intermediate_latent_grid = np.zeros((1, self.latent_depth, h * ls, w * ls))

        for y in range(self.height):
            for x in range(self.width):
                self.intermediate_latent_grid[:, :, y * ls: (y + 1) * ls, x * ls: (x + 1) * ls] = intermediate_latents[y * w + x, :, pad: pad + ls, pad:pad + ls]

        return self.calculate_output_image(self.intermediate_latent_grid, update_all=True)

    def calculate_output_image(self, intermediate_latent_grid, start=(0, 0), end=(0, 0), update_all=False):
        sh = list(intermediate_latent_grid.shape)
        grid_h = sh[2]
        grid_w = sh[3]

        chunk_shape = 24  # has to be multiple of 4
        chunk_stride = chunk_shape // 2
        chunk_overlap = chunk_stride // 2

        output_size = 2 ** (self.output_resolution - self.merge_level)
        if self.output is None or update_all:
            self.output = np.zeros((3, output_size * grid_h, output_size * grid_w), np.uint8)
            start = (0, 0)
            end = (grid_h, grid_w)
            print(f'Updating grid from {start} to {end}.')

        def round_up(number, multiple):
            return int((int((number + multiple - 1) / multiple)) * multiple)

        with tf.Session(graph=tf.Graph()) as session:
            with open(self.network_file, 'rb') as file:
                network = pickle.load(file)
                size = network.output_shape[-1]
                channels = network.output_shape[1]
                self.output_resolution = int(np.log2(size))
                kwargs = {
                    'in_res': self.merge_level + 1,
                    'out_res': self.output_resolution,
                    'latent_depth': int(DEPTHS[self.merge_level - 1]),
                    'latentSize': [None, int(DEPTHS[self.merge_level - 1]), 2 ** self.merge_level, 2 ** self.merge_level],
                    'label_size': 0,
                    'num_channels': channels,
                    'resolution': size
                }
                gsb = network.clone_and_update("GsB", kwargs=kwargs, func='networks.G_new')

                if gsb.input_shape[1:] != [sh[1], chunk_shape, chunk_shape]:
                    print(f'Updating network graph for different input shape {sh[1]}x{chunk_shape}x{chunk_shape}.')
                    gsb.update_latent_size(chunk_shape, chunk_shape)
                    print(f'\tDone!')

                latent_chunk = np.zeros((sh[0], sh[1], chunk_shape, chunk_shape))

                start_y = (start[0] // chunk_stride) * chunk_stride
                start_x = (start[1] // chunk_stride) * chunk_stride

                last_y = max(round_up(end[0], chunk_stride) - chunk_shape, 0)
                last_x = max(round_up(end[1], chunk_stride) - chunk_shape, 0)

                for y in range(start_y, last_y + 1, chunk_stride):
                    for x in range(start_x, last_x + 1, chunk_stride):
                        latent_chunk.fill(0)

                        chunk_h = min(y + chunk_shape, grid_h) - y
                        chunk_w = min(x + chunk_shape, grid_w) - x

                        latent_chunk[:, :, :chunk_h, :chunk_w] = intermediate_latent_grid[:, :, y:y + chunk_h, x:x + chunk_w]
                        try:
                            _, chunk_output = gsb.run_with_session(
                                session,
                                latent_chunk,
                                in_res=self.merge_level + 1,
                                out_res=self.output_resolution,
                                latent_size=[None, sh[1], chunk_shape, chunk_shape],
                                latent_depth=sh[1],
                                minibatch_size=MINIBATCH_SIZE,
                                num_gpus=1,
                                out_mul=127.5,
                                out_add=127.5,
                                out_dtype=np.uint8
                            )
                            chunk_output = np.squeeze(chunk_output)
                        except ValueError:
                            print('Error evaluating GsB')
                            print(f'\t> Latent chunk shape: {latent_chunk.shape}')
                            print(f'\t> GsB input shape: {gsb.input_shape}')

                        ml = 0 if x == 0 else chunk_overlap
                        mt = 0 if y == 0 else chunk_overlap
                        mr = 0 if x >= grid_w - chunk_shape else chunk_overlap
                        mb = 0 if y >= grid_h - chunk_shape else chunk_overlap
                        osz = output_size

                        paste_region = chunk_output[:, mt * osz:(chunk_h - mb) * osz, ml * osz:(chunk_w - mr) * osz]
                        self.output[:, (y + mt) * osz: (y + chunk_h - mb) * osz, (x + ml) * osz: (x + chunk_w - mr) * osz] = paste_region

                self.output = np.squeeze(self.output)

        session.close()
        self._save_instance()
        return self.output

    def calculate_unmerged_output_image(self, intermediate_latent_grid, merge_size=0):
        if merge_size == 0:
            merge_size = self.latent_size

        grid_h = (self.height * self.latent_size) // merge_size
        grid_w = (self.width * self.latent_size) // merge_size

        intermediate_latent_list = np.zeros((grid_h * grid_w, self.latent_depth, merge_size, merge_size))
        sh = list(intermediate_latent_list.shape)
        for y in np.arange(grid_h):
            for x in np.arange(grid_w):
                intermediate_latent_list[y * grid_w + x, :, :, :] = intermediate_latent_grid[:, :, y * merge_size: (y + 1) * merge_size, x * merge_size: (x + 1) * merge_size]

        with tf.Session(graph=tf.Graph()) as session:
            with open(self.network_file, 'rb') as file:
                network = pickle.load(file)
                size = network.output_shape[-1]
                channels = network.output_shape[1]
                self.output_resolution = int(np.log2(size))
                kwargs = {
                    'in_res': self.merge_level + 1,
                    'out_res': self.output_resolution,
                    'latent_depth': int(DEPTHS[self.merge_level - 1]),
                    'latentSize': [None, int(DEPTHS[self.merge_level - 1]), 2 ** self.merge_level, 2 ** self.merge_level],
                    'label_size': 0,
                    'num_channels': channels,
                    'resolution': size
                }
                gsc = network.clone_and_update("GsC", kwargs=kwargs, func='networks.G_new')

                if [sh[1], merge_size, merge_size] != gsc.input_shape[1:]:
                    gsc.update_latent_size(merge_size, merge_size)
                _, outputs = gsc.run_with_session(
                    session,
                    intermediate_latent_list,
                    in_res=self.merge_level + 1,
                    out_res=self.output_resolution,
                    latent_size=[None, sh[1], merge_size, merge_size],
                    latent_depth=sh[1],
                    minibatch_size=MINIBATCH_SIZE,
                    num_gpus=1,
                    out_mul=127.5,
                    out_add=127.5,
                    out_dtype=np.uint8
                )

                output_size = outputs.shape[2]

                output = np.zeros((3, output_size * grid_h, output_size * grid_w), np.uint8)

                for y in range(grid_h):
                    for x in range(grid_w):
                        output[:, y * output_size: (y + 1) * output_size, x * output_size: (x + 1) * output_size] = outputs[y * grid_w + x]

        session.close()
        self._save_instance()
        return np.squeeze(output)

    def get_upsampled(self, image):
        t = self.t_size
        ls = self.latent_size

        if image.shape[2] > 3:
            image = image[:, :, :3]

        img_h = image.shape[0]
        img_w = image.shape[1]

        channels = 3 if len(image.shape) > 2 else 1

        descriptor_size = t * t * channels

        total_pad = (2 ** self.merge_level) - ls
        total_img_pad = total_pad * (t // (2 ** self.merge_level))
        g_size = t - total_img_pad

        self.height = (img_h - total_img_pad) // g_size
        self.width = (img_w - total_img_pad) // g_size

        self.guidance_image = image[:self.height * g_size, :self.width * g_size, :]

        tile_descriptors = np.zeros((self.height * self.width, descriptor_size))
        self.cluster_grid = np.zeros((self.height * ls, self.width * ls), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                tile = image[y * g_size: y * g_size + t, x * g_size: x * g_size + t, :]
                tile_descriptors[y * self.width + x, :] = np.ravel(tile)

        all_indices, all_distances = self.ann_numbers.knn_query(tile_descriptors, 3)

        latent_list = []
        for i in range(self.height * self.width):
            indices = all_indices[i, :]
            random_idx = np.random.randint(len(indices))
            idx = int(indices[random_idx])
            latent_list.append(self.latent_lookup[idx])
            y = i // self.width
            x = i % self.width
            self.cluster_grid[y * ls: (y + 1) * ls, x * ls: (x + 1) * ls] = int(self.cluster_lookup[idx])

        self.latents = np.asarray(latent_list)

        self.intermediate_latents = self.calculate_intermediate_latents(self.latents)
        self.get_output_from_intermediate_latents(self.intermediate_latents)

        grid_h = self.height * ls
        grid_w = self.width * ls

        return self.output, None, (grid_h, grid_w, ls, self.merge_level), 0

    def randomize_grid(self, width, height):
        self.guidance_image = None
        self.randomize_latents(width, height, repeat=False)
        self.intermediate_latents = self.calculate_intermediate_latents(self.latents)
        self.get_output_from_intermediate_latents(self.intermediate_latents)

        self._save_instance()

        return self.output, (self.intermediate_latent_grid.shape[2], self.intermediate_latent_grid.shape[3], self.latent_size, self.merge_level), 0

    def perturb_latent(self, pos_x, pos_y, source_x, source_y, alpha, random_latent=False, from_samples=False, use_cdf=True):
        ls = self.latent_size
        pad = self.get_left_padding()
        grid_h = self.intermediate_latent_grid.shape[2]
        grid_w = self.intermediate_latent_grid.shape[3]

        co = ls // 2

        source_x = min(max(co, source_x), grid_w - (ls - co))
        source_y = min(max(co, source_y), grid_h - (ls - co))
        intermediate_latent = self.intermediate_latent_grid[:, :, source_y - co:source_y - co + ls, source_x - co:source_x - co + ls]

        if random_latent:
            cluster_idx = 0
            if from_samples and self.cluster_samples is not None:
                latent = self.cluster_samples[cluster_idx]
            else:
                cluster = self.latent_clusters[cluster_idx]
                if use_cdf:
                    cdf = self.latent_cdfs[cluster_idx]
                    random_idx = bisect(cdf, np.random.random())
                else:
                    random_idx = np.random.randint(len(cluster))

                random_latent_idx = cluster[random_idx]
                latent = self.latent_lookup[random_latent_idx]

                random_intermediate_latent = self.calculate_intermediate_latents(latent)
                intermediate_latent = random_intermediate_latent[:, :, pad:pad + ls, pad:pad + ls]

        x_start = max(pos_x - co, 0)
        y_start = max(pos_y - co, 0)
        x_end = min(pos_x - co + ls, grid_w)
        y_end = min(pos_y - co + ls, grid_h)

        l = abs(min(0, pos_x - co))
        t = abs(min(0, pos_y - co))
        r = max(grid_w, pos_x - co + ls) - grid_w
        b = max(grid_h, pos_y - co + ls) - grid_h

        assert not r < 0 and not b < 0

        bg = self.intermediate_latent_grid[:, :, y_start:y_end, x_start:x_end]
        fg = intermediate_latent[:, :, t:ls - b, l: ls - r]
        self.intermediate_latent_grid[:, :, y_start:y_end, x_start:x_end] = (1 - alpha) * bg + alpha * fg

        roi = 2 * ls

        self.calculate_output_image(
            self.intermediate_latent_grid,
            start=(max(y_start - roi, 0), max(x_start - roi, 0)),
            end=(min(y_end + 2 * roi, grid_h), min(x_end + 2 * roi, grid_w)),
            update_all=False
        )

        self._save_instance()

        return self.output, 0

    def put_latent(self, pos_x, pos_y, cluster_idx, from_samples=False, use_cdf=True):
        ls = self.latent_size
        pad = self.get_left_padding()
        grid_h = self.intermediate_latent_grid.shape[2]
        grid_w = self.intermediate_latent_grid.shape[3]

        co = ls // 2

        x_start = max(pos_x - co, 0)
        y_start = max(pos_y - co, 0)
        x_end = min(pos_x - co + ls, grid_w)
        y_end = min(pos_y - co + ls, grid_h)

        l = abs(min(0, pos_x - co))
        t = abs(min(0, pos_y - co))
        r = max(grid_w, pos_x - co + ls) - grid_w
        b = max(grid_h, pos_y - co + ls) - grid_h

        assert not r < 0 and not b < 0

        if from_samples and self.cluster_samples is not None:
            random_latent = self.cluster_samples[cluster_idx]
            self.cluster_grid[y_start:y_end, x_start:x_end] = cluster_idx
        else:
            cluster = self.latent_clusters[cluster_idx]
            if use_cdf:
                cdf = self.latent_cdfs[cluster_idx]
                random_idx = bisect(cdf, np.random.random())
            else:
                random_idx = np.random.randint(len(cluster))

            random_latent_idx = cluster[random_idx]
            random_latent = self.latent_lookup[random_latent_idx]
            self.cluster_grid[y_start:y_end, x_start:x_end] = self.cluster_lookup[random_latent_idx]

        random_intermediate_latent = self.calculate_intermediate_latents(random_latent)
        random_intermediate_latent = random_intermediate_latent[:, :, pad:pad + ls, pad:pad + ls]

        if self.latents is not None:
            self.latents[(y_start // ls) * self.width + (x_start // ls)] = random_latent

        self.intermediate_latent_grid[:, :, y_start:y_end, x_start:x_end] = random_intermediate_latent[:, :, t:ls - b, l:ls - r]
        roi = 2 * ls
        self.calculate_output_image(
            self.intermediate_latent_grid,
            start=(max(y_start - roi, 0), max(x_start - roi, 0)),
            end=(min(y_end + 2 * roi, grid_h), min(x_end + 2 * roi, grid_w)),
            update_all=False
        )

        self._save_instance()
        return self.output, 0

    def paste_latents(self, sample_latent, target_x, target_y, target_w, target_h, source_x, source_y, mode='identical'):
        ls = self.latent_size
        pad = self.get_left_padding()
        grid_h = self.intermediate_latent_grid.shape[2]
        grid_w = self.intermediate_latent_grid.shape[3]

        c_offset = ls // 2

        source_x = min(max(c_offset, source_x), grid_w - (ls - c_offset))
        source_y = min(max(c_offset, source_y), grid_h - (ls - c_offset))

        source_latent = self.intermediate_latent_grid[:, :, source_y - c_offset: source_y - c_offset + ls, source_x - c_offset: source_x - c_offset + ls]
        num_similar = 5

        if mode == 'similar':
            image = Image.fromarray(sample_latent)
            image.thumbnail((self.t_size, self.t_size))
            source_descriptor = np.asarray(image)

            indices, distances = self.ann_numbers.knn_query(np.ravel(source_descriptor).reshape(1, -1), num_similar)
            nearest_latents = self.latent_lookup[indices]
            similar_latents = self.calculate_intermediate_latents(np.squeeze(nearest_latents))
        elif mode == 'cluster':
            cluster_idx = self.cluster_grid[source_y, source_x]
            cluster = self.latent_clusters[cluster_idx]
            cdf = self.latent_cdfs[cluster_idx]

        for x in range(0, target_w, ls):
            for y in range(0, target_h, ls):
                x_start = target_x + x
                y_start = target_y + y
                x_end = min(x_start + ls, target_x + target_w)
                y_end = min(y_start + ls, target_y + target_h)

                if mode == 'similar':
                    random_idx = np.random.randint(num_similar)
                    source_latent = similar_latents[random_idx:random_idx + 1, :, pad:pad + ls, pad:pad + ls]
                elif mode == 'cluster':
                    random_idx = bisect(cdf, np.random.random())
                    same_cluster_latent = self.latent_lookup[cluster[random_idx]]
                    intermediate_latent = self.calculate_intermediate_latents(same_cluster_latent)
                    source_latent = intermediate_latent[:, :, pad:pad + ls, pad:pad + ls]
                    self.cluster_grid[y_start:y_end, x_start:x_end] = cluster_idx

                r = max((target_x + target_w), x_start + ls) - (target_x + target_w)
                b = max((target_y + target_h), y_start + ls) - (target_y + target_h)
                self.intermediate_latent_grid[:, :, y_start:y_end, x_start:x_end] = source_latent[:, :, :ls - b, :ls - r]

        roi = 2 * ls
        self.calculate_output_image(
            self.intermediate_latent_grid,
            start=(max(target_y - roi, 0), max(target_x - roi, 0)),
            end=(min(target_y + target_h + 2 * roi, grid_h), min(target_x + target_w + 2 * roi, grid_w)),
            update_all=False
        )

        self._save_instance()
        return self.output, 0

    def set_merge_level(self, level, latent_size=-1):
        self.merge_level = level
        if latent_size < 0:
            self.latent_size = 2 ** level
        else:
            self.latent_size = latent_size

        self.latent_depth = int(DEPTHS[self.merge_level - 1])

        print(f'New level: {self.merge_level}, decrusted width: {self.latent_size}')

        self._save_instance()

    def get_dominant_cluster_colors(self):
        if not self.dominant_cluster_colors:
            for image in self.average_images:
                pixels = np.float32(image.reshape(-1, 3))
                clustering = KMeans(n_clusters=5).fit(pixels)
                count = np.bincount(clustering.labels_)
                sorted_indices = np.argsort(count)[::-1]
                sorted_colors = clustering.cluster_centers_[sorted_indices, :]
                self.dominant_cluster_colors.append(sorted_colors[0].astype(np.uint8))

        self._save_instance()
        return self.dominant_cluster_colors

    def get_cluster_output(self):
        dominant_colors = self.get_dominant_cluster_colors()
        cluster_output = np.zeros((self.height * self.latent_size, self.width * self.latent_size, 3), dtype=np.uint8)
        for y in range(self.height * self.latent_size):
            for x in range(self.width * self.latent_size):
                cluster_output[y, x, :] = dominant_colors[self.cluster_grid[y, x]]

        self._save_instance()
        return cluster_output

    def get_output(self):
        self.calculate_output_image(self.intermediate_latent_grid)
        return self.output, (self.intermediate_latent_grid.shape[2], self.intermediate_latent_grid.shape[3], self.latent_size, self.latent_size)

    def get_unmerged_output(self):
        return self.calculate_unmerged_output_image(self.intermediate_latent_grid)

    def get_latent_images(self):
        return self.latent_images

    def get_latent_averages(self):
        return self.average_images

    def get_left_padding(self):
        return (2 ** self.merge_level - self.latent_size) // 2

    def get_cluster_at(self, source_y, source_x):
        return self.cluster_grid[source_y, source_x]

    def save_latents(self):
        grid_h = self.intermediate_latent_grid.shape[2]
        grid_w = self.intermediate_latent_grid.shape[3]
        time_str = time.strftime("%m_%d_%H%M")
        np.save(str(Path.home()) + f'\\Desktop\\{self.selected_dataset}_{grid_h}x{grid_w}_latents_{time_str}', self.intermediate_latent_grid)

    def load_latents(self, path):
        self.intermediate_latent_grid = np.load(path)
        self.calculate_output_image(self.intermediate_latent_grid, update_all=True)
        self._save_instance()
        return self.output, (self.intermediate_latent_grid.shape[2], self.intermediate_latent_grid.shape[3], self.latent_size, self.merge_level), 0


class Server(BaseManager):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    port = 8080
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()

    manager = TileGanManager()

    server_process = Server(address=('', port), authkey=b'tilegan')
    # server_process.register('sampleFromCluster', manager.sampleFromCluster)
    server_process.register('find_available_datasets', manager.find_available_datasets)
    server_process.register('load_dataset', manager.load_dataset)
    server_process.register('get_latent_images', manager.get_latent_images)
    server_process.register('get_latent_averages', manager.get_latent_averages)
    server_process.register('get_dominant_cluster_colors', manager.get_dominant_cluster_colors)
    server_process.register('get_output', manager.get_output)
    server_process.register('get_unmerged_output', manager.get_unmerged_output)
    server_process.register('get_cluster_output', manager.get_cluster_output)
    server_process.register('get_cluster_at', manager.get_cluster_at)
    server_process.register('get_upsampled', manager.get_upsampled)
    server_process.register('put_latent', manager.put_latent)
    server_process.register('perturb_latent', manager.perturb_latent)
    server_process.register('paste_latents', manager.paste_latents)
    server_process.register('save_latents', manager.save_latents)
    server_process.register('load_latents', manager.load_latents)
    # server_process.register('improveLatents', manager.MRFLatents)
    server_process.register('set_merge_level', manager.set_merge_level)
    server_process.register('randomize_grid', manager.randomize_grid)
    # server_process.register('deadLeaves', manager.deadLeaves)
    # server_process.register('undo', manager.undo)

    server = server_process.get_server()
    print(f'Server is listening... Connect application to {ip}:{port}.')
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        print('Bye.')
