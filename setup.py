from setuptools import find_packages, setup


# Collect packages
packages = find_packages(exclude=('tests', 'experiments'))
print('Found the following packages to be created:\n  {}'.format('\n  '.join(packages)))

# Setup the package
setup(
    name='oodd',
    version='1.0.0',
    packages=packages,
    python_requires='>=3.8.0',
    url='https://github.com/JakobHavtorn/hvae-oodd',
    author='Jakob D. Havtorn',
)
