# setup.py

from skbuild import setup

# It's good practice to read the long description from a README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    # THIS IS THE CRITICAL FIX:
    # We must explicitly define the package name here.
    name="PyCSEMRI",

    version="0.1.0",
    description="A package for CSEMRI with C++ accelerated components.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    author="Daiki Tamada",
    author_email="dtamada@wisc.edu",
    url="https://github.com/your-username/PyCSEMRI_clean",
    license="GPL",
    
    packages=["package"],
    # This line is no longer needed with scikit-build and a flat structure
    # package_dir={"": "package"}, 
    cmake_source_dir=".",
    
    install_requires=[
        "numpy<1.20; python_version == '3.6'",
        "numpy; python_version >= '3.7'",
    ],
    
    python_requires=">=3.6",
)

