from setuptools import setup, find_packages

setup(
    name="lit_ras",
    version="0.1.0",
    author="Jonathan Ipe and Bryce Tu Chi",
    author_email="your.email@example.com",
    description="A brief description of your package",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your-repository",
    packages=find_packages(),
    # install_requires=[
    #     # List your package dependencies here
    #     "numpy",
    #     "scipy",
    #     "torch",
    # ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)