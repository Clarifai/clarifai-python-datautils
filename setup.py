import re
import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

with open("./clarifai_datautils/__init__.py") as f:
  content = f.read()
_search_version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content)
assert _search_version
version = _search_version.group(1)

install_requires = [
    'unstructured[pdf] @ git+https://github.com/clarifai/unstructured.git@support_clarifai_model',
    'llama-index-core==0.10.33',
    'llama-index-llms-clarifai==0.1.2',
    'pi_heif==0.18.0'
]

packages = setuptools.find_namespace_packages(include=["clarifai_datautils*"])

setuptools.setup(
    name="clarifai-datautils",
    version=f"{version}",
    author="Clarifai",
    author_email="support@clarifai.com",
    description="Clarifai Data Utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clarifai/clarifai-python-datautils",
    packages=packages,
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
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
    extras_require={
        'annotations': ["datumaro==1.6.1"],
    },
    include_package_data=True)
