#!/usr/bin/env python

import unittest

import nbconvert

import numpy as np


with open("assignment16.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)

with open("assignment16.py", "w") as f:
    f.write(python_file)

from assignment16 import MotionDetector


class TestSolution(unittest.TestCase):

    def test_difference_array(self):

        md = MotionDetector('pumpjack.mp4')
        md.create_image_difference(rank=1)
        gold_array = np.load('pumpjack_image_array_gold.npy', allow_pickle=True)
        np.testing.assert_array_equal(md.images_matrix[:1000, :1000], gold_array)
        del md


if __name__ == '__main__':
    unittest.main()
