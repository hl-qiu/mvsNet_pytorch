import importlib


# find the dataset definition by name, for example dtu_yao (dtu_yao.py)
# 按名称查找数据集定义，例如dtu_yao (dtu_yao.py)
def find_dataset_def(dataset_name):
    # default='dtu_yao'
    module_name = 'datasets.{}'.format(dataset_name)
    # 导入模块：module_name = datasets.dtu_yao
    # importlib.import_module(name, package=None)
    # name： 1、绝对路径导入，例如name=pkg.mod
    #        2、相对路径导入，例如name='..mod'，此时需要定义package参数，package='pkg.subpkg'
    module = importlib.import_module(module_name)
    # 获取module中MVSDataset，存在就打印出来
    return getattr(module, "MVSDataset")
