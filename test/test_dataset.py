from unittest import TestCase

from dataset import Dataset
import torch


class TestDataset(TestCase):

    def setUp(self):
        super().setUp()
        path_to_data_dir = '../data'
        self._train_dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
        self._test_dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TEST)

    def test___len__(self):
        self.assertEqual(len(self._train_dataset), 60000)
        self.assertEqual(len(self._test_dataset), 10000)

    def test___getitem__(self):
        image, label = self._train_dataset[0]
        self.assertEqual(image.shape, torch.Size((1, 28, 28)))
        self.assertAlmostEqual(image.sum().item(), 17.761684, delta=1e-6)
        self.assertEqual(label.item(), 5)

        image, label = self._test_dataset[100]
        self.assertEqual(image.shape, torch.Size((1, 28, 28)))
        self.assertAlmostEqual(image.sum().item(), -101.310890, delta=1e-6)
        self.assertEqual(label.item(), 6)
