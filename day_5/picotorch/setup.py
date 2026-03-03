import setuptools

setuptools.setup(
    name="picotorch",
    version="0.2.0",
    author="Vignesh Yaadav",
    author_email="vigneshyadav27@gmail.com",
    description="A tiny tensor-based autograd engine with CPU and CUDA backends, inspired by micrograd",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.20",
        "cffi>=1.15",
    ],
    extras_require={
        "cuda": ["cupy-cuda12x"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
