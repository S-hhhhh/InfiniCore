import os
import sys
import subprocess
from set_env import set_env

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
os.chdir(PROJECT_DIR)

def run_tests(test_names, device_type):
    failed = []
    base_dir = os.path.join(PROJECT_DIR, "test", "infiniop-test")

    for test in test_names:

        # GGUF 生成阶段
        cmd_gguf = f"python -m test_generate.testcases.{test}"
        res = subprocess.run(
            cmd_gguf,
            shell=True,
            cwd=base_dir
        )
        if res.returncode != 0:
            print(f"[ FAIL ] GGUF generation failed for {test}")
            failed.append(test)
            continue

        # C++ 测试阶段
        exe = os.path.join(PROJECT_DIR, "build/linux/x86_64/release/infiniop-test")
        gguf_path = os.path.join(base_dir, f"{test}.gguf")
        cmd_cpp = f"{exe} {gguf_path} {device_type} --warmup 20 --run 1000"
        res = subprocess.run(
            cmd_cpp,
            shell=True,
            cwd=PROJECT_DIR
        )
        if res.returncode != 0:
            print(f"[ FAIL ] C++ test failed for {test}")
            failed.append(test)

    return failed

if __name__ == "__main__":

    set_env()
    test_names = ["add", "gemm", "mul", "random_sample", "swiglu", "clip"]
    failed = run_tests(test_names, " ".join(sys.argv[1:]))

    if len(failed) == 0:
        print("\033[92mAll tests passed!\033[0m")
    else:
        print("\033[91mThe following tests failed:\033[0m")
        for t in failed:
            print(f" - {t}")
    sys.exit(len(failed))
