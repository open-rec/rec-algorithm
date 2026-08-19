from setuptools import setup, find_packages

setup(
    name='rec-algorithm',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    author='xsank',
    author_email='xsank@foxmail.com',
    extras_require={
        'spark': ['pyspark==3.5.3'],
        'publish': ['redis>=5,<9', 'elasticsearch>=8,<9'],
        'cluster': ['pyspark==3.5.3', 'redis>=5,<9', 'elasticsearch>=8,<9'],
    },
    entry_points={
        'console_scripts': [
            'openrec-spark-recall=jobs.spark.recall_job:main',
        ],
    },
)
