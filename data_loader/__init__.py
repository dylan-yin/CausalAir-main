# from data_loader.data_loaders import MnistDataLoader
# from data_loader.airdata_loader import AirdataLoader
from data_loader.pyg_loader import pygdataLoader
from data_loader.pygmm_loader import pygmmdataLoader
from data_loader.sts_loader import stsdataLoader
from data_loader.sts_loader_bj import stsdataLoader_bj

__all__ =[
    # AirdataLoader,
    # MnistDataLoader,
    pygdataLoader,
    stsdataLoader,
    stsdataLoader_bj,

]