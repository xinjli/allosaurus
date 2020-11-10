import unittest
from pathlib import Path
import requests

class TestModel(unittest.TestCase):

    def test_latest_available(self):
        req = requests.head('https://www.pyspeech.com/static/model/recognition/allosaurus/latest.tar.gz')
        self.assertTrue(req.status_code==200)

if __name__ == '__main__':
    unittest.main()