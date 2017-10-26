from setuptools import setup, find_packages

setup(
    name="pymp2rage",
    version="0.0.1dev",
    long_description=__doc__,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    author='Gilles de Hollander',
    keywords='MRI high-resolution laminar',
    install_requires=['numpy', 'nibabel', 'nilearn'],
)
