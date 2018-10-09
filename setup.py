from setuptools import setup

setup(name='mlperf',
      version='0.1a1',
      description='Machine Learning Performance Assessor',
      long_description='Machine Learning Performance Assessor',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development',
      ],
      python_requires='~=3.6',
      keywords='ml machinelearning research performance',
      url='https://github.com/v-m/ml-perf',
      author='Vincenzo Musco',
      author_email='muscovin@gmail.com',
      license='MIT',
      packages=['mlperf', 'mlperf.clustering', 'mlperf.clustering.gaussianmixture', 'mlperf.clustering.hierarchical',
                'mlperf.clustering.kmeans', 'mlperf.tools'],
      install_requires=['numpy', 'pandas', 'scipy'],
      scripts=['bin/generate-clusters', 'bin/convert-sparse-dense'],
      include_package_data=True,
      zip_safe=False)
