from setuptools import setup


setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='mypackage',
    url='https://github.com/yannholzer/mypackage',
    author='Yannick Eyholzer',
    author_email='yannick.eyholzer@unige.ch',
    # Needed to actually package something
    packages=["lightcurve"],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='A python package consisting in useful function for my phd',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)