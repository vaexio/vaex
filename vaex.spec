# -*- mode: python -*-
import vaex
a = Analysis(['bin/vaex'],
             pathex=['/net/theon/data/users/breddels/vaex/src/SubspaceFinding'],
             hiddenimports=["h5py.h5ac", "six"],
             hookspath=None,
             runtime_hooks=None)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name=vaex.__program_name__,
          debug=False,
          strip=None,
          upx=True,
          console=True )
          
icon_tree = Tree('python/vaex/ui/icons', prefix = 'icons')
data_tree = Tree('data', prefix='data')
doc_tree = Tree('doc', prefix='doc')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               icon_tree,
               doc_tree,
               data_tree,
               strip=None,
               upx=True,
               name=vaex.__clean_name__)
