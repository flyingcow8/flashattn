import subprocess
import os
import shutil

def build_with_cmake(source_dir, build_dir, install_dir=None, cmake_args=None, build_args=None):
    """
    Compile cpp file with CMake

    参数:
        source_dir (str): C++ 项目的源代码目录
        build_dir (str): 构建目录
        install_dir (str, 可选): 安装目录，如果需要安装的话
        cmake_args (list, 可选): 传递给 CMake 的额外参数
        build_args (list, 可选): 传递给构建工具的额外参数
    """
    os.makedirs(build_dir, exist_ok=True)

    # config CMake
    cmake_command = [
        "cmake",
        source_dir,
        "-DCMAKE_BUILD_TYPE=Release"
    ]
    if cmake_args:
        cmake_command.extend(cmake_args)

    try:
        subprocess.check_call(cmake_command, cwd=build_dir)
    except subprocess.CalledProcessError as e:
        print(f"CMake config failed: {e}")
        return False

    # build
    build_command = [
        "cmake",
        "--build",
        build_dir,
        "--config",
        "Release"
    ]
    if build_args:
        build_command.extend(build_args)

    try:
        subprocess.check_call(build_command)
    except subprocess.CalledProcessError as e:
        print(f"build failed: {e}")
        return False

    # install
    if install_dir:
        install_command = [
            "cmake",
            "--install",
            build_dir,
            "--prefix",
            install_dir
        ]
        try:
            subprocess.check_call(install_command)
        except subprocess.CalledProcessError as e:
            print(f"install failed: {e}")
            return False

    return True


if __name__ == "__main__":

    kernel_path = "./build_kernel/kernels"
    full_kernel_path = "./build_kernel/full_kernels"

    if os.path.exists(kernel_path) and os.path.exists(full_kernel_path):
        print(f"{kernel_path} and {full_kernel_path} is exists, skip run generator.")
    else:
        print("run kernel file generator...")
        result = subprocess.run(['bash', "./run_generator.sh"], capture_output=False, text=True)
        if not result:
            print("generate kernel file failed")
            sys.exist(-1)

    # source_directory = "../.."
    # build_directory = "build/capi/"
    # install_directory = "install"

    # MACA_PATH = os.getenv('MACA_PATH')
    # if MACA_PATH is None:
    #     raise KeyError("Environment variable 'MACA_PATH' not found.")
    # print(f"MACA_PATH:{MACA_PATH}")

    # # build
    # success = build_with_cmake(
    #     source_dir=source_directory,
    #     build_dir=build_directory,
    #     install_dir=None,
    #     cmake_args=["-DBUILD_WITH_CPP=false",f"-DMACA_PATH={MACA_PATH}"],
    #     build_args=["--parallel", "112"]
    # )

    # if success:
    #     print("build kernel success!")
    # else:
    #     print("build kernel failed.")
