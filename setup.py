from distutils.core import setup
from distutils.extension import Extension

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
        name="nifti",
        version="0.0.1",
        ext_modules=[
            Extension(
                name='nifti',
                sources=['src/' + f for f in ['niftilib.c',
                                              'nifti1_io.c', 'znzlib.c']],
                define_macros=[("HAVE_ZLIB", None)],
                include_dirs=['/usr/local/include'],
                libraries=['z'],
                library_dirs=['/usr/local/lib'],
                extra_compile_args=["-O3", "-march=native"]
            )
        ]
    )
