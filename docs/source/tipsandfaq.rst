Tips and FAQ
============

Which shortcut keys are available
---------------------------------

All shortcuts can be found in the menubar, for instance to select the lasso tool, look at the menubar at Mode->Lasso, on OS X it is Cmd-L.


How do I read file X in format Y
--------------------------------

The best format to use is hdf5, if you have any other format, check if TOPCAT can read it (http://www.star.bris.ac.uk/~mbt/topcat/). From TOPCAT you can export to FITS (choose colfits). Now you can open the file in vaex. However, FITS stores the data in big endian format, while most computer now (x86(-64) instruction set) use low endian format. For better performance you can export to hdf5 format from vaex again.

Converting to fits can also be done from the command line, using TOPCAT/STILTS:

.. code:: python

	topcat -stilts tcopy ofmt=colfits-plus yourdata.asc yourdata.fits

What is this shuffling about when exporting?
--------------------------------------------

If you export a dataset, vaex asks if you want to shuffle it. What this means is that the order of the data (the rows) are random after exporting. This can be useful when you dataset is large that plotting becomes slow (for instance working on a laptop). Now you can choose to show only a fraction of the data (for instance 1%), and still get a good idea what is in the data.


Can I have a X and Y in the plot
--------------------------------

If you want extra things in your plot, fancy axes, highlighting something, overplotting extra data etc you should choose ``File->Export data/script``. This will generate a Python script and extra files to reproduce the plot you see, and allows you to edit the script. 