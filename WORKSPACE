workspace(name = "galvASR")

local_repository(
    name = "org_tensorflow",
    path = "third_party/tensorflow",
)

new_local_repository(
    name = "kaldi",
    path = "third_party/kaldi/",
    build_file = "kaldi.BUILD",
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(path_prefix = "", tf_repo_name = "org_tensorflow")

# cuda_configure(name="local_config_cuda")
# tensorrt_configure(name="local_config_tensorrt")

# load("@org_tensorflow//third_party/gpus:cuda_configure.bzl", "cuda_configure")
# load("@org_tensorflow//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")
# load("@org_tensorflow//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")