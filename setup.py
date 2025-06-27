# setup.py

from skbuild import setup

# It's good practice to read the long description from a README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    version="0.1.0", # `version` is now defined here
    description="A package for CSEMRI with C++ accelerated components.", # `description` is here
    long_description=long_description, # `readme` content is here
    long_description_content_type="text/markdown",
    
    # These were not declared as dynamic, so they stay here
    author="Daiki Tamada",
    author_email="dtamada@wisc.edu",
    url="https://github.com/your-username/PyCSEMRI_clean",
    license="GPL",
    
    packages=["package"],
    cmake_source_dir=".",
    
    # These dependencies correspond to the `dynamic` "dependencies" field
    install_requires=[
        "numpy<1.20; python_version == '3.6'",
        "numpy; python_version >= '3.7'",
    ],
    
    # This corresponds to the `dynamic` "requires-python" field
    python_requires=">=3.6",
)

