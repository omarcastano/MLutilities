from setuptools import setup, find_packages

setup(
    name='MLutilities',
    version='1.0.0',
    author='Omar Andres Castaño',
    description='Machine Learning Utilities',
    long_description='This packages provides a series of machine learning utilities which makes easy to teach machine learning topics',
    url='https://github.com/omarcastano/MLutilities.git',
    packages=find_packages()
    package_data = {'MLutilities':['datasets/diamonds.pkl']}
)
