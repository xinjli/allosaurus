import unittest
from pathlib import Path
from allosaurus.app import read_recognizer

class TestRecognition(unittest.TestCase):

    def test_latest_nonempty(self):
        audio_file = Path(__file__).parent.parent / 'sample.wav'
        model = read_recognizer('latest')
        results = model.recognize(audio_file)
        self.assertTrue(len(results) > 0)

    def test_eng_nonempty(self):
        audio_file = Path(__file__).parent.parent / 'sample.wav'
        model = read_recognizer('latest')
        results = model.recognize(audio_file, 'eng')
        self.assertTrue(len(results) > 0)

if __name__ == '__main__':
    unittest.main()