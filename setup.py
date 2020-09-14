import setuptools

cpp_ext = setuptools.Extension(
    name="eventprop.lif_layer_cpp",
    sources=["src/lif_layer_cpp.cpp"],
    include_dirs=["src/pybind11/include", "src/eigen"],
    language="c++",
    extra_compile_args=["-std=c++11", "-Ofast"],
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
    data_files=[
        (yin_yang_dir, "test_labels.npy"),
        (yin_yang_dir, "train_labels.npy"),
        (yin_yang_dir, "valid_labels.npy"),
        (yin_yang_dir, "test_samples.npy"),
        (yin_yang_dir, "train_samples.npy"),
        (yin_yang_dir, "valid_samples.npy"),
    ],
)
