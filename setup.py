# setup.py

from skbuild import setup

# It's good practice to read the long description from a README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PyCSEMRI",
    version="0.1.0",
    author="Daiki Tamada",
    author_email="dtamada@wisc.edu",
    description="A package for CSEMRI with C++ accelerated components.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dtamadauw/PyCSEMRI_fast",
    license="GPL",
    
    packages=["package"],
    package_dir={"": "package"},
    cmake_source_dir=".",
    
    # Use environment markers to specify version-dependent dependencies
    install_requires=[
        "numpy<1.20; python_version == '3.6'",
        "numpy; python_version >= '3.7'",
    ],
    
    python_requires=">=3.6",
)

