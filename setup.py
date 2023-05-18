from setuptools import setup

setup(
   name='allosaurus',
   version='2.0.1',
   description='A python speech recognition framework',
   author='Xinjian Li',
   package_data={'': ['*.yml', '*.csv']},
   install_requires=[
      'scipy',
      'numpy',
      'torch',
      'editdistance',
      'loguru'
   ]
)
