import glob
import os
import os.path as osp
import platform
import sys

from setuptools import find_packages, setup

__version__ = None
exec(open("abysssplat/version.py", "r").read())

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)


def get_extensions():
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("abysssplat", "cuda", "csrc")
    sources = glob.glob(osp.join(extensions_dir, "*.cu")) + glob.glob(
        osp.join(extensions_dir, "*.cpp")
    )
    sources = [path for path in sources if "hip" not in path]

    undef_macros = []
    define_macros = []
    if sys.platform == "win32":
        define_macros.append(("abysssplat_EXPORTS", None))

    extra_compile_args = {"cxx": ["-O3"]}
    if os.name != "nt":
        extra_compile_args["cxx"].append("-Wno-sign-compare")
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"].append("-DAT_PARALLEL_OPENMP")
        extra_compile_args["cxx"].append("/openmp" if sys.platform == "win32" else "-fopenmp")
    else:
        print("Compiling without OpenMP...")

    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"].extend(["-arch", "arm64"])
        extra_link_args.extend(["-arch", "arm64"])

    nvcc_flags = os.getenv("NVCC_FLAGS", "").split()
    nvcc_flags.extend(["-O3", "--use_fast_math"])
    if LINE_INFO:
        nvcc_flags.append("-lineinfo")
    if torch.version.hip:
        define_macros.append(("USE_ROCM", None))
        undef_macros.append("__HIP_NO_HALF_CONVERSIONS__")
    else:
        nvcc_flags.append("--expt-relaxed-constexpr")
    if sys.platform == "win32":
        nvcc_flags.append("-DWIN32_LEAN_AND_MEAN")
    extra_compile_args["nvcc"] = nvcc_flags

    return [
        CUDAExtension(
            "abysssplat.csrc",
            sources,
            include_dirs=[osp.join(extensions_dir, "third_party", "glm")],
            define_macros=define_macros,
            undef_macros=undef_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]


setup(
    name="abysssplat",
    version=__version__,
    description="Active-illumination-aware Gaussian splatting for deep-sea 3D reconstruction",
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=find_packages(),
    include_package_data=True,
)
