from setuptools import setup, find_packages

setup(
    name="jax-model-zoo",
    version="0.1.0",
    packages=find_packages(),
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-timeout>=2.1.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "diffusers>=0.20.0",
            "timm>=0.9.0",
            "scipy>=1.9.0",
        ],
        "torch": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "diffusers>=0.20.0",
            "timm>=0.9.0",
        ],
        "checkpoint": [
            "orbax-checkpoint>=0.5.0",
            "msgpack>=1.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-timeout>=2.1.0",
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "transformers>=4.30.0",
            "diffusers>=0.20.0",
            "timm>=0.9.0",
            "orbax-checkpoint>=0.5.0",
            "msgpack>=1.0.0",
            "scipy>=1.9.0",
        ],
    },
)
