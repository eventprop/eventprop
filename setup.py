import setuptools

cpp_ext = setuptools.Extension(
    name="eventprop.eventprop_cpp",
    sources=["src/eventprop_cpp.cpp"],
    include_dirs=[
        "src/pybind11/include",
        "src/eigen",
        "src/boost_format/include",
        "src/boost_math/include",
    ],
    language="c++",
    # extra_compile_args=["-std=c++17", "-O0", "-g"]
    extra_compile_args=[
        "-std=c++17",
        "-O3",
        "-march=native",
        "-fopenmp",
        "-freciprocal-math",
        "-fno-signed-zeros",
        "-funroll-loops",
    ],
)

yin_yang_dir = "eventprop/yin_yang_data_set/publication_data"

setuptools.setup(
    name="eventprop",
    version="0.1",
    author="Timo C. Wunderlich",
    packages=["eventprop"],
    python_requires=">=3.8",
    install_requires=["numpy", "scipy", "dask"],
    ext_modules=[cpp_ext],
    include_package_data=True,
)
