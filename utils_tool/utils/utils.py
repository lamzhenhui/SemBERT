

def check_cuda():
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 'mps'

    # 检查是否有可用的NVIDIA GPU
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def keras_version_check_v2():
    """
    检查keras版本是否大于2.0
    """
    from keras import __version__ as keras_version
    from packaging import version

    # 获取Keras版本
    # print("Keras Version:", keras_version)

    # 检查Keras版本是否大于2

    if version.parse(keras_version) > version.parse('2'):
        # print("Keras版本大于2")
        return True
        # 使用Keras 2.x的API
    else:
        # print("Keras版本不大于2")
        return False


def version_check(pack_name='', vesion_lg_parse=''):
    import importlib
    from packaging import version

    def get_module_version(pack_name):
        module = importlib.import_module(pack_name)
        return module.__version__
    # from pack_name import __version__ as pac_version

    # 获取Keras版本
    # print("Keras Version:", keras_version)

    # 检查Keras版本是否大于2

    print(get_module_version(pack_name))
    if version.parse(get_module_version(pack_name)) > version.parse(vesion_lg_parse):
        return True
        # 使用Keras 2.x的API
    else:
        # print("Keras版本不大于2")
        return False


if __name__ == '__main__':
    # keras_version_check_v2()
    print(version_check(pack_name='scipy', vesion_lg_parse='1.14'))
    print(1)
