from setuptools import setup, find_packages

setup(
    name="spectra-ml",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask>=2.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "plotly>=5.3.0",
        "umap-learn>=0.5.0",
        "scipy>=1.7.0",
        "werkzeug>=2.0.0",
        "openpyxl>=3.0.0",
        "markdown>=3.3.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
        ],
    },
    author="Anshuman Nayak",
    author_email="anshuman@example.com",
    description="A machine learning platform for spectral data analysis",
    keywords="machine learning, spectra, data analysis",
    url="https://github.com/anshuman-nayak/spectra-ml",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
