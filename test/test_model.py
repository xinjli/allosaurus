import unittest
from pathlib import Path
import requests


class TestModel(unittest.TestCase):

    def test_latest_available(self):
        model_name = "latest"
        url = 'https://github.com/xinjli/allosaurus/releases/download/v1.0/' + model_name + '.tar.gz'
        req = requests.head(url)
        print(req.status_code)
        self.assertTrue(req.status_code == 200 or req.status_code == 302)


if __name__ == '__main__':
    unittest.main()
