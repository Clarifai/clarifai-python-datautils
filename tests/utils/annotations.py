import os.path as osp


def get_asset_path(*args):
  cur_dir = osp.dirname(__file__)
  return osp.abspath(osp.join(cur_dir, "..", "assets", *args))
