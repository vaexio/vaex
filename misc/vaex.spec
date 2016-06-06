# -*- mode: python -*-
import vaex
import astropy
import os
astropy_base_dir = os.path.dirname(astropy.__file__)


a = Analysis(['bin/vaex'],
             pathex=['/net/theon/data/users/breddels/vaex/src/SubspaceFinding'],
             hiddenimports=["h5py.h5ac", "six", "PyQt4", "PyQt4.QtGui", "PyQt4.QtCore", "PyQt4.QtTest"],
             hookspath=None,
             runtime_hooks=["python/vaex/ui/rthook_pyqt4.py"])
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name=vaex.__build_name__, #+"_app",
          debug=False,
          strip=None,
          upx=True,
          console=True )
          
icon_tree = Tree('python/vaex/ui/icons', prefix = 'icons')
data_tree = Tree('data', prefix='data', excludes=["*.properties"])
#doc_tree = Tree('doc', prefix='doc', excludes=["*.zip"])
astropy_tree = Tree(astropy_base_dir, prefix='astropy')

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               icon_tree,
               data_tree,
               astropy_tree,
               strip=None,
               upx=True,
               name=vaex.__build_name__)
