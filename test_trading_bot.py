import unittest
from utils.utils import init_dask_client, cleanup_dask_client, setup_logging
from dask.distributed import Client
import logging

class TestUtils(unittest.TestCase):

    def test_init_dask_client(self):
        client = init_dask_client()
        self.assertIsInstance(client, Client)  # Check if the returned object is an instance of Client
        cleanup_dask_client(client)

    def test_cleanup_dask_client(self):
        client = init_dask_client()
        cleanup_dask_client(client)
        self.assertTrue(client.status == 'closed')  # Check if the client is closed

    def test_setup_logging(self):
        setup_logging()
        logger = logging.getLogger()
        self.assertEqual(logger.level, logging.DEBUG)  # Check if the logging level is set to DEBUG

if __name__ == '__main__':
    unittest.main()

