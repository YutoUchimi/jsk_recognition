#!/usr/bin/env python

import argparse
import multiprocessing

import jsk_data


def download_data(*args, **kwargs):
    p = multiprocessing.Process(
        target=jsk_data.download_data,
        args=args,
        kwargs=kwargs)
    p.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='quiet', action='store_false')
    args = parser.parse_args()
    quiet = args.quiet

    PKG = 'jsk_perception'

    download_data(
        pkg_name=PKG,
        path='learning_datasets/kitchen_dataset.tgz',
        url='https://drive.google.com/uc?id=1iBSxX7I0nFDJfYNpFEb1caSQ0nl4EVUa',
        md5='61d6fcda7631fd0a940bc751c8f21031',
        quiet=quiet,
        extract=True,
    )

    download_data(
        pkg_name=PKG,
        path='learning_datasets/human_size_mirror_dataset.tgz',
        url='https://drive.google.com/uc?id=1MJAvGKr_IUoX_PK9GGwkQerxHrta84av',
        md5='addcd2da612b506859723e6ece34ddb3',
        quiet=quiet,
        extract=True,
    )


if __name__ == '__main__':
    main()
