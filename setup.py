from setuptools import setup, find_packages

setup(
    name="multi-layer-price-prediction",
    version="1.0.0",
    description="Institutional-grade multi-layer probability-based price prediction system",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "yfinance>=0.2.28",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "statsmodels>=0.14.0",
        "arch>=6.2.0",
        "torch>=2.0.0",
        "transformers>=4.33.0",
        "fastapi>=0.103.0",
        "uvicorn>=0.23.0",
        "vaderSentiment>=3.3.2",
        "feedparser>=6.0.10",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0", "pytest-cov>=4.1.0"],
    },
)
