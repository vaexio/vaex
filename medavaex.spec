# -*- mode: python -*-
a = Analysis(['bin/medavaex'],
             pathex=['/net/theon/data/users/breddels/gavi/src/SubspaceFinding'],
             hiddenimports=["h5py.h5ac", "six"],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='medavaex',
          debug=False,
          strip=None,
          upx=True,
          console=True )
          
icon_tree = Tree('python/gavi/icons', prefix = 'icons')          
data_tree = Tree('data/dist', prefix='data')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               icon_tree,
               data_tree,
               strip=None,
               upx=True,
               name='medavaex')
