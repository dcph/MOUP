from setuptools import setup, find_packages

setup(
        name='offlinerlmoup',
        version="0.0.1",
        description=(
            'OfflineRL-moup'
        ),
        packages=find_packages(),
        platforms=["all"],
        install_requires=[
            "d4rl",
            "gym",
            "matplotlib",
            "numpy",
            "pandas",
            "ray",
            "torch",
            "tensorboard",
            "tqdm",
        ]
    )
