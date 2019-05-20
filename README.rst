|Travis| |Conda| |Chat| 

Vaex uses several sites:

* Main page: https://vaex.io/
* Documentation: https://docs.vaex.io/
* Github: https://github.com/vaexio/vaex
* PyPi: https://pypi.python.org/pypi/vaex/


Vaex is open source software, if you need support, contact us at https://vaex.io



What is Vaex?
-------------

Vaex is a python library for lazy **Out-of-Core DataFrames** (similar to
Pandas), to visualize and explore big tabular datasets. It can calculate
*statistics* such as mean, sum, count, standard deviation etc, on an
*N-dimensional grid* for more than **a billion** (:math:`10^9`) objects/rows
**per second**. Visualization is done using **histograms**, **density
plots** and **3d volume rendering**, allowing interactive exploration of
big data. Vaex uses memory mapping, zero memory copy policy and lazy
computations for best performance (no memory wasted).

Why vaex
========

-  **Performance:** Works with huge tabular data, process
   more than a *billion* rows/second
-  **Lazy / Virtual columns:** compute on the fly, without wasting ram
-  **Memory efficient** no memory copies when doing
   filtering/selections/subsets.
-  **Visualization:** directly supported, a one-liner is often enough.
-  **User friendly API:** You will only need to deal with a Dataset
   object, and tab completion + docstring will help you out:
   ``ds.mean<tab>``, feels very similar to Pandas.
-  **Lean:** separated into multiple packages

   -  ``vaex-core``: Dataset and core algorithms, takes numpy arrays as
      input columns.
   -  ``vaex-hdf5``: Provides memory mapped numpy arrays to a Dataset.
   -  ``vaex-arrow``: `Arrow <https://arrow.apache.org/>`__ support for
      cross language data sharing.
   -  ``vaex-viz``: Visualization based on matplotlib.
   -  ``vaex-jupyter``: Interactive visualization based on Jupyter
      widgets / ipywidgets, bqplot, ipyvolume and ipyleaflet.
   -  ``vaex-astro``: Astronomy related transformations and FITS file
      support.
   -  ``vaex-server``: Provides a server to access a dataset remotely.
   -  ``vaex-distributed``: (Proof of concept) combined multiple servers
      / cluster into a single dataset for distributed computations.
   -  ``vaex-qt``: Program written using Qt GUI.
   -  ``vaex``: meta package that installs all of the above.
   -  ``vaex-ml``: `Machine learning <http://docs.vaex.io/en/latest/ml.html>`__ with automatic pipelines.

-  **Jupyter integration**: vaex-jupyter will give you interactive
   visualization and selection in the Jupyter notebook and Jupyter lab.

Installation
------------

Using conda:

-  ``conda install -c conda-forge vaex``

Using pip:

-  ``pip install vaex``

Or read the `detailed instructions <https://docs.vaex.io/en/latest/installing.html>`__

Getting started
===============

We assuming you have installed vaex, and are running a `Jupyter notebook
server <https://jupyter.readthedocs.io/en/latest/running.html>`__. We
start by importing vaex and ask it to give us sample example dataset.

.. code:: ipython3

    import vaex
    ds = vaex.example()  # open the example dataset provided with vaex


Instead, you can `download some larger datasets <https://docs.vaex.io/en/latest/datasets.html>`__, or
`read in your csv file <https://docs.vaex.io/en/latest/api.html#vaex.from_csv>`__.

.. code:: ipython3

    ds  # will pretty print a table




.. raw:: html

    <table>
    <thead>
    <tr><th>#     </th><th>x           </th><th>y           </th><th>z           </th><th>vx         </th><th>vy         </th><th>vz         </th><th>E              </th><th>L                 </th><th>Lz                 </th><th>FeH                </th></tr>
    </thead>
    <tbody>
    <tr><td>0     </td><td>-0.777470767</td><td>2.10626292  </td><td>1.93743467  </td><td>53.276722  </td><td>288.386047 </td><td>-95.2649078</td><td>-121238.171875 </td><td>831.0799560546875 </td><td>-336.426513671875  </td><td>-2.309227609164518 </td></tr>
    <tr><td>1     </td><td>3.77427316  </td><td>2.23387194  </td><td>3.76209331  </td><td>252.810791 </td><td>-69.9498444</td><td>-56.3121033</td><td>-100819.9140625</td><td>1435.1839599609375</td><td>-828.7567749023438 </td><td>-1.788735491591229 </td></tr>
    <tr><td>2     </td><td>1.3757627   </td><td>-6.3283844  </td><td>2.63250017  </td><td>96.276474  </td><td>226.440201 </td><td>-34.7527161</td><td>-100559.9609375</td><td>1039.2989501953125</td><td>920.802490234375   </td><td>-0.7618109022478798</td></tr>
    <tr><td>3     </td><td>-7.06737804 </td><td>1.31737781  </td><td>-6.10543537 </td><td>204.968842 </td><td>-205.679016</td><td>-58.9777031</td><td>-70174.8515625 </td><td>2441.724853515625 </td><td>1183.5899658203125 </td><td>-1.5208778422936413</td></tr>
    <tr><td>4     </td><td>0.243441463 </td><td>-0.822781682</td><td>-0.206593871</td><td>-311.742371</td><td>-238.41217 </td><td>186.824127 </td><td>-144138.75     </td><td>374.8164367675781 </td><td>-314.5353088378906 </td><td>-2.655341358427361 </td></tr>
    <tr><td>...   </td><td>...         </td><td>...         </td><td>...         </td><td>...        </td><td>...        </td><td>...        </td><td>...            </td><td>...               </td><td>...                </td><td>...                </td></tr>
    <tr><td>329995</td><td>3.76883793  </td><td>4.66251659  </td><td>-4.42904139 </td><td>107.432999 </td><td>-2.13771296</td><td>17.5130272 </td><td>-119687.3203125</td><td>746.8833618164062 </td><td>-508.96484375      </td><td>-1.6499842518381402</td></tr>
    <tr><td>329996</td><td>9.17409325  </td><td>-8.87091351 </td><td>-8.61707687 </td><td>32.0       </td><td>108.089264 </td><td>179.060638 </td><td>-68933.8046875 </td><td>2395.633056640625 </td><td>1275.490234375     </td><td>-1.4336036247720836</td></tr>
    <tr><td>329997</td><td>-1.14041007 </td><td>-8.4957695  </td><td>2.25749826  </td><td>8.46711349 </td><td>-38.2765236</td><td>-127.541473</td><td>-112580.359375 </td><td>1182.436279296875 </td><td>115.58557891845703 </td><td>-1.9306227597361942</td></tr>
    <tr><td>329998</td><td>-14.2985935 </td><td>-5.51750422 </td><td>-8.65472317 </td><td>110.221558 </td><td>-31.3925591</td><td>86.2726822 </td><td>-74862.90625   </td><td>1324.5926513671875</td><td>1057.017333984375  </td><td>-1.225019818838568 </td></tr>
    <tr><td>329999</td><td>10.5450506  </td><td>-8.86106777 </td><td>-4.65835428 </td><td>-2.10541415</td><td>-27.6108856</td><td>3.80799961 </td><td>-95361.765625  </td><td>351.0955505371094 </td><td>-309.81439208984375</td><td>-2.5689636894079477</td></tr>
    </tbody>
    </table>



Using `square brackets[] <https://docs.vaex.io/en/latest/api.html#vaex.dataset.Dataset.__getitem__>`__,
we can easily filter or get different views on the dataset.

.. code:: ipython3

    ds_negative = ds[ds.x < 0]  # easily filter your dataset, without making a copy
    ds_negative[:5][['x', 'y']]  # take the first five rows, and only the 'x' and 'y' column (no memory copy!)




.. raw:: html

    <table>
    <thead>
    <tr><th style="text-align: right;">  #</th><th style="text-align: right;">         x</th><th style="text-align: right;">       y</th></tr>
    </thead>
    <tbody>
    <tr><td style="text-align: right;">  0</td><td style="text-align: right;"> -0.777471</td><td style="text-align: right;"> 2.10626</td></tr>
    <tr><td style="text-align: right;">  1</td><td style="text-align: right;"> -7.06738 </td><td style="text-align: right;"> 1.31738</td></tr>
    <tr><td style="text-align: right;">  2</td><td style="text-align: right;"> -5.17174 </td><td style="text-align: right;"> 7.82915</td></tr>
    <tr><td style="text-align: right;">  3</td><td style="text-align: right;">-15.9539  </td><td style="text-align: right;"> 5.77126</td></tr>
    <tr><td style="text-align: right;">  4</td><td style="text-align: right;">-12.3995  </td><td style="text-align: right;">13.9182 </td></tr>
    </tbody>
    </table>



When dealing with huge datasets, say a billion rows (:math:`10^9`),
computations with the data can waste memory, up to 8 GB for a new
column. Instead, vaex uses lazy computation, only a representation of
the computation is stored, and computations done on the fly when needed.
Even though, you can just many of the numpy functions, as if it was a
normal array.

.. code:: ipython3

    import numpy as np
    # creates an expression (nothing is computed)
    r = np.sqrt(ds.x**2 + ds.y**2 + ds.z**2)
    r  # for convenience, we print out some values




.. parsed-literal::

    <vaex.expression.Expression(expressions='sqrt((((x ** 2) + (y ** 2)) + (z ** 2)))')> instance at 0x11bcc4780 values=[2.9655450396553587, 5.77829281049018, 6.99079603950256, 9.431842752707537, 0.8825613121347967 ... (total 330000 values) ... 7.453831761514681, 15.398412491068198, 8.864250273925633, 17.601047186042507, 14.540181524970293] 



These expressions can be added to the dataset, creating what we call a
*virtual column*. These virtual columns are simular to normal columns,
except they do not waste memory.

.. code:: ipython3

    ds['r'] = r  # add a (virtual) column that will be computed on the fly
    ds.mean(ds.x), ds.mean(ds.r)  # calculate statistics on normal and virtual columns




.. parsed-literal::

    (-0.06713149126400597, 9.407082338299773)



One of the core features of vaex is its ability to calculate statistics
on a regular (N-dimensional) grid. The dimensions of the grid are
specified by the binby argument (analogous to SQL's grouby), and the
shape and limits.

.. code:: ipython3

    ds.mean(ds.r, binby=ds.x, shape=32, limits=[-10, 10]) # create statistics on a regular grid (1d)




.. parsed-literal::

    array([15.01058183, 14.43693006, 13.72923338, 12.90294499, 11.86615103,
           11.03563695, 10.12162553,  9.2969267 ,  8.58250973,  7.86602644,
            7.19568442,  6.55738773,  6.01942499,  5.51462457,  5.15798991,
            4.8274218 ,  4.7346551 ,  5.1343761 ,  5.46017944,  6.02199777,
            6.54132124,  7.27025256,  7.99780777,  8.55188217,  9.30286584,
            9.97067561, 10.81633293, 11.60615795, 12.33813552, 13.10488982,
           13.86868565, 14.60577266])



.. code:: ipython3

    ds.mean(ds.r, binby=[ds.x, ds.y], shape=32, limits=[-10, 10]) # or 2d
    ds.count(ds.r, binby=[ds.x, ds.y], shape=32, limits=[-10, 10]) # or 2d counts/histogram




.. parsed-literal::

    array([[22., 33., 37., ..., 58., 38., 45.],
           [37., 36., 47., ..., 52., 36., 53.],
           [34., 42., 47., ..., 59., 44., 56.],
           ...,
           [73., 73., 84., ..., 41., 40., 37.],
           [53., 58., 63., ..., 34., 35., 28.],
           [51., 32., 46., ..., 47., 33., 36.]])



These one and two dimensional grids can be visualized using any plotting
library, such as matplotlib, but the setup can be tedious. For
convenience we can use `plot1d <https://docs.vaex.io/en/latest/api.html#vaex.dataset.Dataset.plot1d>`__,
`plot <https://docs.vaex.io/en/latest/api.html#vaex.dataset.Dataset.plot>`__, or see the `list of
plotting commands <https://docs.vaex.io/en/latest/api.html#visualization>`__



Continue
--------

`Continue the tutorial <https://docs.vaex.io/en/latest/tutorial.html>`__ or check the
`examples <https://docs.vaex.io/en/latest/examples.html>`__

If you like vaex, please let us know by giving us a star on GitHub,

Regards,

The vaex.io team

.. |Travis| image:: https://travis-ci.org/vaexio/vaex.svg?branch=master
   :target: https://travis-ci.org/vaexio/vaex.svg?branch=master
.. |Chat| image:: https://badges.gitter.im/maartenbreddels/vaex.svg
   :alt: Join the chat at https://gitter.im/maartenbreddels/vaex
   :target: https://gitter.im/maartenbreddels/vaex?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Conda| image:: https://anaconda.org/conda-forge/vaex/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/vaex   
   
