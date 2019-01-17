#!/usr/bin/env python

import argparse

import mvtk

from jsk_recognition_utils.datasets import SemanticSegmentationDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root_dir', required=True)
    parser.add_argument('-a', '--aug', action='store_true')
    args = parser.parse_args()

    dataset = SemanticSegmentationDataset(args.root_dir, aug=args.aug)
    mvtk.datasets.view_dataset(dataset, SemanticSegmentationDataset.visualize)


if __name__ == '__main__':
    main()
