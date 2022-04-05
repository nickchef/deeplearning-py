from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(name="deeplearning-py",
      version="0.1",
      packages=find_packages(),
      license='MIT',
      author='Nicholas Lyu',
      author_email="zlyu0226@uni.sydney.edu.au",
      install_requires=['numpy', 'scipy'],
      python_requires=">=3.8",
      long_description=long_description,
      long_description_content_type="text/markdown",
      )
