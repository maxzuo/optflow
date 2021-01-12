import setuptools
import sys

setuptools.setup(
    name="optflow",
    version="0.0.1-beta1",
    author="Max Zuo",
    author_email="max.zuo@gmail.com",
    description="Wrapper convenience methods for OpenCV optical flow",
    long_description="None",
    url="https://github.com/maxzuo/optflow",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib', 'scikit-learn'],
    extras_require={
        'full': ['tqdm', 'opencv-python>=4.3', 'opencv-contrib-python>=4.3']
    },
    entry_points={
        'cv2': ['.optflow = optflow']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)