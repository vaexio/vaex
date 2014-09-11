# -*- mode: python -*-
import gavi.vaex
a = Analysis(['bin/vaex'],
             pathex=['/net/theon/data/users/breddels/gavi/src/SubspaceFinding'],
             hiddenimports=["h5py.h5ac", "six"],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name=gavi.vaex.__program_name__,
          debug=False,
          strip=None,
          upx=True,
          console=True )
          
icon_tree = Tree('python/gavi/icons', prefix = 'icons')          
data_tree = Tree('data/dist', prefix='data')
html_tree = Tree('html', prefix='html')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               icon_tree,
               html_tree,
               data_tree,
               strip=None,
               upx=True,
               name=gavi.vaex.__full_name__)
