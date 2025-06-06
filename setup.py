from setuptools import setup, find_packages

setup(
    name="echogains",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "coremltools",
        "numpy",
        "onnx",
        "onnxruntime",
        "pillow",
    ],
)
