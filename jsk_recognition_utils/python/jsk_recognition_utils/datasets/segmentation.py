from __future__ import print_function

import os
import os.path as osp

import chainer
try:
    import chainer_mask_rcnn as cmr
except ImportError:
    print('chainer_mask_rcnn cannot be imported.')
import cv2
import imgaug.augmenters as iaa
import mvtk
import numpy as np


class SemanticSegmentationDataset(chainer.dataset.DatasetMixin):

    def __init__(self, root_dir, aug=False):
        self.root_dir = root_dir
        self.aug = aug

        class_names_path = osp.join(root_dir, 'class_names.txt')
        with open(class_names_path, 'r') as f:
            class_names = f.readlines()
        self.class_names = [name.rstrip() for name in class_names]

        self._images = []
        self._labels = []
        images_dir = osp.join(root_dir, 'JPEGImages')
        labels_dir = osp.join(root_dir, 'SegmentationClass')
        for image_ in sorted(os.listdir(images_dir)):
            image_path = osp.join(images_dir, image_)
            basename = image_.rstrip('.jpg')
            label_path = osp.join(labels_dir, basename + '.npy')
            self._images.append(image_path)
            self._labels.append(label_path)

    def __len__(self):
        return len(self._images)

    def get_example(self, i):
        image_path = self._images[i]
        label_path = self._labels[i]

        image = cv2.imread(image_path)
        image = image[:, :, ::-1]  # BGR -> RGB
        assert image.dtype == np.uint8
        assert image.ndim == 3

        label = np.load(label_path)
        assert label.dtype == np.int32
        assert label.ndim == 2

        # Data augmentation
        if self.aug:
            def st(x):
                return iaa.Sometimes(0.3, x)

            # 1. Color augmentation
            obj_datum = dict(img=image)
            random_state = np.random.RandomState()
            augs = [
                st(iaa.Add([-50, 50], per_channel=True)),
                st(iaa.InColorspace(
                    'HSV', children=iaa.WithChannels(
                        [1, 2], iaa.Multiply([0.5, 2])))),
                st(iaa.GaussianBlur(sigma=[0.0, 1.0])),
                st(iaa.AdditiveGaussianNoise(
                    scale=(0.0, 0.1 * 255), per_channel=True)),
            ]
            obj_datum = next(mvtk.aug.augment_object_data(
                [obj_datum], random_state=random_state, augmentations=augs))
            image = obj_datum['img']

            # 2. Geometric augmentation
            # 2-1. Affine transform
            obj_datum2 = dict(img=image, lbl=label)
            random_state2 = np.random.RandomState()
            augs = [
                st(iaa.Affine(scale=(0.7, 1.5), order=0,
                              mode=['constant', 'symmetric'])),
                st(iaa.Affine(translate_percent=(-0.1, 0.1),
                              mode=['constant', 'symmetric'])),
                st(iaa.Affine(rotate=(-45, 45), order=0,
                              mode=['constant', 'symmetric'])),
                st(iaa.Affine(shear=(-20, 20), order=0,
                              mode=['constant', 'symmetric'])),
            ]
            obj_datum = next(mvtk.aug.augment_object_data(
                [obj_datum2], random_state=random_state2, augmentations=augs))
            image = obj_datum2['img']
            label = obj_datum2['lbl']

            # 2-2. Flip
            if np.random.uniform() < 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)
            if np.random.uniform() < 0.5:
                image = np.flipud(image)
                label = np.flipud(label)

        return image, label

    def visualize(self, index):
        image, label = self[index]

        print('[%04d] %s' % (index, '>' * 75))
        print('image shape: %s' % repr(image.shape))
        print('[%04d] %s' % (index, '<' * 75))
        label_viz = mvtk.image.label2rgb(
            label, img=image, label_names=self.class_names, alpha=0.7)
        viz = mvtk.image.tile(
            [image, label_viz], (1, 2))

        return mvtk.image.resize(viz, size=600 * 600)


class InstanceSegmentationDataset(chainer.dataset.DatasetMixin):

    def __init__(self, root_dir):
        self.root_dir = root_dir

        class_names_path = osp.join(root_dir, 'class_names.txt')
        with open(class_names_path, 'r') as f:
            class_names = f.readlines()
        # instance id 0 is '_background_' and should be ignored.
        self.fg_class_names = [name.rstrip() for name in class_names][1:]

        self._images = []
        self._class_labels = []
        self._instance_labels = []
        images_dir = osp.join(root_dir, 'JPEGImages')
        class_labels_dir = osp.join(root_dir, 'SegmentationClass')
        instance_labels_dir = osp.join(root_dir, 'SegmentationObject')
        for image_ in sorted(os.listdir(images_dir)):
            image_path = osp.join(images_dir, image_)
            basename = image_.rstrip('.jpg')
            class_label_path = osp.join(
                class_labels_dir, basename + '.npy')
            instance_label_path = osp.join(
                instance_labels_dir, basename + '.npy')
            self._images.append(image_path)
            self._class_labels.append(class_label_path)
            self._instance_labels.append(instance_label_path)

    def __len__(self):
        return len(self._images)

    def get_example(self, i):
        image_path = self._images[i]
        class_label_path = self._class_labels[i]
        instance_label_path = self._instance_labels[i]

        image = cv2.imread(image_path)
        assert image.dtype == np.uint8
        assert image.ndim == 3

        class_label = np.load(class_label_path)
        assert class_label.dtype == np.int32
        assert class_label.ndim == 2

        instance_label = np.load(instance_label_path)
        # instance id 0 is '_background_' and should be ignored.
        instance_label[instance_label == 0] = -1
        assert instance_label.dtype == np.int32
        assert instance_label.ndim == 2

        assert image.shape[:2] == class_label.shape == instance_label.shape

        labels, bboxes, masks = cmr.utils.label2instance_boxes(
            label_instance=instance_label, label_class=class_label,
            return_masks=True,
        )
        masks = masks.astype(np.int32, copy=False)
        labels = labels.astype(np.int32, copy=False)
        labels -= 1  # background: 0 -> -1
        bboxes = bboxes.astype(np.float32, copy=False)

        return image, bboxes, labels, masks
