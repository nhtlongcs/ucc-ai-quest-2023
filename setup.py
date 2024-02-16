from setuptools import setup

setup(
    name='ucc-core',
    version='0.1',
    package_dir={"core": "."},
    install_requires=[
        # List any dependencies your CLI may have
    ],
    entry_points={
        'console_scripts': [
            'ucc-train=cli.train:main',
            'ucc-eval=cli.validate:main',
            'ucc-eval-fast=cli.validate_batch:main',
            'ucc-pred=cli.predict:main',
        ],
    },
)