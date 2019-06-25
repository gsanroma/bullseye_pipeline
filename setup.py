#!/usr/bin/env python

"""
pipeline for creating bullseye representation from FreeSurfer output
"""
import os
if os.path.exists('MANIFEST'): os.remove('MANIFEST')


def main(**extra_args):
    from setuptools import setup
    setup(name='bullseye_pipeline',
          version='1.0.0',
          description='bullseye pipeline',
          long_description="""creates bullseye parcellation of white matter.""" + \
          """It requires (part of) FreeSurfer output.""",
          author= 'Gerard Sanroma-Guell',
          author_email='gsanroma@gmail.com',
          packages = ['bullseye_pipeline'],
          entry_points={
            'console_scripts': [
                             "run_bullseye_pipeline=bullseye_pipeline.run_bullseye_pipeline:main"
                              ]
                       },
          license='BSD',
          classifiers = [c.strip() for c in """\
            Development Status :: 1 - Beta
            Intended Audience :: Developers
            Intended Audience :: Science/Research
            Operating System :: OS Independent
            Programming Language :: Python
            Topic :: Software Development
            """.splitlines() if len(c.split()) > 0],    
          maintainer = 'Gerard Sanroma-Guell',
          maintainer_email = 'gsanroma@gmail.com',
          install_requires=["nipype","nibabel"],
          **extra_args
         )

if __name__ == "__main__":
    main()

