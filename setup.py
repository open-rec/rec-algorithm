from setuptools import setup, find_packages

setup(
    name='rec-algorithm',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={'algorithm.feature.definitions': ['*.json']},
    author='xsank',
    author_email='xsank@foxmail.com',
    extras_require={
        'lightgbm': ['lightgbm>=4.3,<5'],
        'spark': ['pyspark==4.0.4'],
        'publish': ['redis>=5,<9', 'elasticsearch>=8,<9'],
        'cluster': [
            'pyspark==4.0.4', 'redis>=5,<9', 'elasticsearch>=8,<9',
            'pydantic>=2.7,<3', 'lightgbm>=4.3,<5',
        ],
    },
    entry_points={
        'console_scripts': [
            'openrec-spark-recall=jobs.spark.recall_job:main',
        ],
    },
)
