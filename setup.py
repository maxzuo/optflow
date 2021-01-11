import setuptools
import sys

def get_requirements():
    requirements = ['numpy', 'matplotlib', 'scikit-learn']
    # check if opencv is built from source and exists:
    if sys.version_info < (3,0):
        print("Package does not support python2", file=sys.stderr)
        sys.exit(1)

    import pkgutil, importlib
    if pkgutil.find_loader('cv2') is None and importlib.util.find_spec('cv2') is None:
        requirements.append('opencv-python>=4.3')
        requirements.append('opencv-contrib-python>=4.3')
    print("REQ", requirements)
    return requirements

setuptools.setup(
    name="optflow",
    version="0.0.1-beta",
    author="Max Zuo",
    author_email="max.zuo@gmail.com",
    description="Wrapper convenience methods for OpenCV optical flow",
    long_description="None",
    url="https://github.com/maxzuo/optflow",
    packages=setuptools.find_packages(),
    install_requires=get_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ]
)