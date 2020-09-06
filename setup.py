import setuptools

cpp_ext = setuptools.Extension(
    name = "eventprop.lif_layer_cpp",
    sources = ["src/lif_layer_cpp.cpp"],
    include_dirs = ["src/pybind11/include", "src/eigen"],
    language = "c++",
    extra_compile_args = ["-std=c++11", "-Ofast"]
)

setuptools.setup(
    name = "eventprop",
    version = "0.1",
    author = "Timo C. Wunderlich",
    packages = ["eventprop"],
    python_requires = ">=3.8",
    install_requires = ["numpy", "scipy", "dask"],
    ext_modules = [cpp_ext],
)