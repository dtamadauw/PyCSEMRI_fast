# setup.py

from skbuild import setup

# It's good practice to read the long description from a README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pycsemri",
    version="0.1.8", # It's good practice to bump the version for new changes
    description="A package for CSEMRI with C++ accelerated components.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    author="Daiki Tamada",
    author_email="dtamada@wisc.edu",
    url="https://github.com/your-username/PyCSEMRI_fast",
    license="GPL",
    
    # This now points to your new directory name
    packages=["pycsemri"],
    
    cmake_source_dir=".",
    
    # These dependencies correspond to the `dynamic` "dependencies" field
    install_requires=[
        "numpy<1.20; python_version == '3.6'",
        "numpy; python_version >= '3.7'",
    ],
    
    # This corresponds to the `dynamic` "requires-python" field
    python_requires=">=3.6",
)

