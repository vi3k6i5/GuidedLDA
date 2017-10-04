from setuptools import setup, Command
import subprocess
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize


class PyTest(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        errno = subprocess.call(['py.test'])
        raise SystemExit(errno)

name = 'guidedLDA'
version = '1.0'

cmdclass = {'test': PyTest}

try:
    from sphinx.setup_command import BuildDoc
    cmdclass['build_sphinx'] = BuildDoc
except ImportError:
    print('WARNING: sphinx not available, not building docs')

extensions = [Extension("guidedlda/*", ["*.pyx"])]

setup(
    name=name,
    version=version,
    url='http://github.com/vi3k6i5/guidedLDA',
    author='Vikash Singh',
    author_email='vikash.duliajan@gmail.com',
    description='Run Guided LDA so topics are as per requirements.',
    long_description=open('README.rst').read(),
    packages=['guidedlda'],
    install_requires=['numpy'],
    platforms='any',
    cmdclass=cmdclass,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
    ],
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', version)
        }
    },
    ext_modules=cythonize("guidedlda/_guidedlda.pyx")
)
