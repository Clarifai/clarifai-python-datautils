import setuptools

# with open("README.md", "r") as fh:
#   long_description = fh.read()

with open("VERSION", "r") as f:
  version = f.read().strip()

with open("requirements.txt", "r") as fh:
  install_requires = fh.read().split('\n')

if install_requires and install_requires[-1] == '':
  # Remove the last empty line
  install_requires = install_requires[:-1]

packages = setuptools.find_namespace_packages(include=["clarifai_datautils*"])

setuptools.setup(
    name="clarifai-datautils",
    version=f"{version}",
    author="Clarifai",
    author_email="support@clarifai.com",
    description="Clarifai Data Utils",
    long_description='Data utils for Clarifai',
    long_description_content_type="text/markdown",
    url="https://github.com/Clarifai/clarifai-python-datautils",
    packages=packages,
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache 2.0",
    python_requires='>=3.8',
    install_requires=install_requires,
    include_package_data=True)
