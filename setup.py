from setuptools import setup, find_packages

setup(
    name="Python_DataStructure",  
    version="1.0.5",
    author="Mehrdad Hasanzade",
    author_email="mehrdadt75m@gmail.com",
    description="A Python package for data structures and algorithms",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)
