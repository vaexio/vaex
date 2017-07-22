
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
    <meta name="description" content="">
    <meta name="author" content="">
    <link rel="icon" href="logos/vaex_alt.png">

    <title>Vaex: Visualization and exploration of big tabular data</title>

    <!-- Bootstrap core CSS -->
    <link href="css/bootstrap.min.css" rel="stylesheet">

    <!-- IE10 viewport hack for Surface/desktop Windows 8 bug -->
    <!-- <link href="../../assets/css/ie10-viewport-bug-workaround.css" rel="stylesheet">-->

    <!-- HTML5 shim and Respond.js for IE8 support of HTML5 elements and media queries -->
    <!--[if lt IE 9]>
      <script src="https://oss.maxcdn.com/html5shiv/3.7.2/html5shiv.min.js"></script>
      <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
    <![endif]-->

    <link href="css/vaex.css" rel="stylesheet">
    <link href="//cdnjs.cloudflare.com/ajax/libs/ekko-lightbox/4.0.1/ekko-lightbox.min.css" rel="stylesheet">
    <link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/font-awesome/4.6.3/css/font-awesome.min.css">
    <link rel="stylesheet" href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.5.0/styles/monokai-sublime.min.css">


  </head>
<!-- NAVBAR
================================================== -->
  <body>
    <div class="navbar-wrapper">
      <div class="">

        <nav class="navbar navbar-inverse navbar-static-top">
          <div class="container">
            <div class="navbar-header">
              <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
                <span class="sr-only">Toggle navigation</span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
              </button>
                <!--<a class="navbar-brand" href="#"><span class="glyphicon icon-spectabular"></span>vaex</a> -->
               <!--  <a class="navbar-brand" href="#"><span class="glyphicon icon-spectabular"></span>V&#xe6;X</a>-->
              <a class="navbar-brand" href="http://vaex.astro.rug.nl"><li class="glyphicon icon-telescope3" style="color: white"></li>vaex</i>(beta)</a>
            </div>
            <div id="navbar" class="navbar-collapse collapse">
              <ul class="nav navbar-nav">
                <li><a href="#download" class="page-scroll" ><i class="fa fa-download"></i>Download</a></li>
                <li class="dropdown">
                  <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false"><i class="fa fa-eye"></i>Demonstration<span class="caret"></span></a>
                  <ul class="dropdown-menu">
                    <li> <a data-width="1025" data-title="Vaex demo" href="https://www.youtube.com/watch?v=NpWLef7-0Yg" data-toggle="lightbox" role="button"><i class="fa fa-film"></i>Demonstration movie</a> </li>
                    <li> <a href="#demo" class="page-scroll"><i class="fa fa-desktop"></i>Live demonstration</a> </li>
                    <li><a href="#example" class="page-scroll"><i class="fa fa-file-code-o"></i>Example code</a></li>
                  </ul>
              </li>
                <li><a href="http://vaex.astro.rug.nl/latest/"><i class="fa fa-book"></i>Documentation</a></li>
                <li class="dropdown">
                  <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false">More <span class="caret"></span></a>
                  <ul class="dropdown-menu">
                    <li><a href="https://github.com/maartenbreddels/vaex"><i class="fa fa-github"></i>Github</a></li>
                    <li><a href="http://vaex.astro.rug.nl/latest/installing.html"><i class="fa fa-download"></i>Download/Install instructions</a></li>
                    <li><a href="https://pypi.python.org/pypi/vaex/"><i class="fa fa-archive"></i>Vaex on pypi</a></li>
                    <li><a href="http://vaex.astro.rug.nl/latest/gallery.html"><i class="fa fa-picture-o"></i>Gallery</a></li>
                    <li role="separator" class="divider"></li>
                    <li class="dropdown-header"><i class="fa fa-book"></i>Documentation</li>
                    <li><a href="http://vaex.astro.rug.nl/latest/"><i class="fa-empty"></i>Home</a></li>
                    <li><a href="http://vaex.astro.rug.nl/latest/tutorial_ipython_notebook.html"><i class="fa-empty"></i>Tutorials</a></li>
                    <li><a href="http://vaex.astro.rug.nl/latest/api.html"><i class="fa "></i>API</a></li>
                  </ul>
                </li>
              </ul>
            </div>
          </div>
        </nav>

      </div>
    </div>


    <!-- Carousel
    ================================================== -->
    <div id="myCarousel" class="carousel slide" data-ride="carousel">
      <!-- Indicators -->
      <ol class="carousel-indicators">
        <li data-target="#myCarousel" data-slide-to="0" class="active"></li>
        <li data-target="#myCarousel" data-slide-to="1"></li>
        <li data-target="#myCarousel" data-slide-to="2"></li>
        <li data-target="#myCarousel" data-slide-to="3"></li>
      </ol>
      <div class="carousel-inner" role="listbox">
        <!--<div class="item active" style="background: url(overview.png) no-repeat left center; background-size: cover;>-->
        <div class="item active" style="background: url(image/aq_nytaxi.png) no-repeat center center; background-size: cover;">
          <!--<img class="first-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="First slide">-->
          <!--<img src="aq_nytaxi.png"> -->
          <div class="container">
            <div class="carousel-caption">
              <h1>Fast visualization of big data.</h1>
                Plot 1 billion points in ~1 second, with interactive navigation on a single computer.
              <!--<p><a href="nyct.png" data-toggle="lightbox" data-title="New York City Taxi pickup locations" data-footer=""  data-gallery="global-gallery" data-parent="" ><img src="nyct.png" width="20%"/></a></p>-->
              <!--<p><a class="btn btn-lg btn-primary" href="#" role="button">Or watch the demo movie</a></p>-->
            </div>
          </div>
        </div>
        <div class="item">
          <img src="image/linked_views.png">
          <!--<img class="second-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Second slide">-->
          <div class="container">
            <div class="carousel-caption alt-color">
              <h1>Interactive selections.</h1>
              <p>Lasso selection, with instant redraw.</p>
              <!--<p><img src="linked_views.png" width="30%"/></a></p>-->
              <!--<p><a class="btn btn-lg btn-primary" href="#" role="button">Learn more</a></p>-->
            </div>
          </div>
        </div>
        <div class="item">
          <img src="image/notebook.png">
          <!--<img class="third-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Third slide">-->
          <div class="container">
            <div class="carousel-caption">
              <h1>Jupyter notebook integration.</h1>
                <p>Interactive navigation and selections are possible</p>
              <!--<p><a href="notebook.png" data-toggle="lightbox" data-title="In a Jupyter notebook" data-footer="Interactive navigation and selection possible in the notebook"  data-gallery="global-gallery" data-parent="" ><img src="notebook.png" width="30%"/></a></p>-->
              <!--<p><a class="btn btn-lg btn-primary" href="#" role="button">Browse gallery</a></p>-->
            </div>
          </div>
        </div>
        <div class="item">
          <img src="image/volr.png"/>
          <!--<img class="fourth-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Second slide">-->
          <div class="container">
            <div class="carousel-caption ">
              <h1>Volume rendering.</h1>
              <p>Vectors overlayed.</p>
              <!--<p><img src="volr.png" width="30%"/></p>-->
              <!--<p><a class="btn btn-lg btn-primary" href="#" role="button">Learn more</a></p>-->
            </div>
          </div>
        </div>
      </div>
      <a class="left carousel-control" href="#myCarousel" role="button" data-slide="prev">
        <span class="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
        <span class="sr-only">Previous</span>
      </a>
      <a class="right carousel-control" href="#myCarousel" role="button" data-slide="next">
        <span class="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
        <span class="sr-only">Next</span>
      </a>
    </div><!-- /.carousel -->


    <!-- Marketing messaging and featurettes
    ================================================== -->
    <!-- Wrap the rest of the page in another container to center all the content. -->
    <div class="jumbotron">
    <h1><li class="glyphicon icon-telescope3"></li>vaex: visualization and exploration of big tabular data</h1>
        <p class="lead">A billion objects per second on a single computer. <span class="text-muted">Standalone or Python library.</span></p>
        <p>
          <a class="btn btn-lg btn-success page-scroll" href="#download" role="button"><i class="fa fa-download"></i>Download</a>
          <a id="bla" class="btn btn-lg btn-success" href="https://www.youtube.com/watch?v=NpWLef7-0Yg" data-toggle="lightbox" data-width="1024" role="button"><i class="fa fa-film"></i>Demo movie</a>
        </p>
    </div>

    <div class="container marketing">

      <!-- Three columns of text below the carousel -->
      <div class="row">
        <div class="col-sm-4">
          <a href="image/nyc_taxi_full.png" data-toggle="lightbox" data-title="New York City Taxi pickup locations" data-footer=""  data-gallery="gallery-1" data-parent="" >
            <img class="img-circle" src="image/nyc_taxi_square.png" alt="Generic placeholder image" width="140" height="140">
          </a>
          <h2>Why use vaex</h2>
          <p>
            Visualize and explore <b>huge tabular datasets interactively</b>...
          </p>
          <p><a class="btn btn-default page-scroll" href="#fast" role="button">Read more &raquo;</a></p>
        </div><!-- /.col-lg-4 -->
        <div class="col-sm-4">
          <a href="image/linked_views.png" data-toggle="lightbox" data-title="Helmi and de Zeeuw 2000" data-footer="Selection made on the right, also visible in the left"  data-gallery="gallery-1" data-parent="" >
            <img class="img-circle" src="image/linked_views.png" alt="Generic placeholder image" width="140" height="140">
          </a>
          <h2>How does it work</h2>
          <p>vaex does this by visualizing binned aggregated data...</p>
          <p><a class="btn btn-default page-scroll" href="#explore" role="button">Read more &raquo;</a></p>
        </div><!-- /.col-lg-4 -->
        <div class="col-sm-4">
          <a href="image/notebook.png" data-toggle="lightbox" data-title="Vaex in the Jupyter notebook" data-footer=""  data-gallery="gallery-1" data-parent="" >
            <img class="img-circle" src="image/notebook.png" alt="Generic placeholder image" width="140" height="140">
          </a>
          <h2>What is vaex</h2>
          <p>A graphical interface, or library that integrates with the Jupyter/IPython notebook...</p>
          <p><a class="btn btn-default page-scroll" href="#flexible" role="button">Read more &raquo;</a></p>
        </div><!-- /.col-lg-4 -->
      </div><!-- /.row -->


      <!-- START THE FEATURETTES -->

      <hr class="featurette-divider">
      <a id="fast" name="fast"></a>
      <div class="row featurette">
        <div class="col-md-8">
          <h2 class="featurette-heading">Why use vaex? <span class="text-muted"></span></h2>
            <p class="lead">
            <ul>
                <li>Visualize and explore <b>big tabular data</b> interactively</li>
                <li>Process more than a <b>billion objects</b> per second on a single computer.</li>
                <li>Custom mathematical expressions can <b>transform</b> data on the fly.</li>
                <li><b>Explore</b> the dataset by using visual queries and boolean expressions to visualize subsets of the data.</li>
                <li>vaex has a <b>graphical interface</b> for most common uses cases.</li>
                <li>vaex is also Python library for custom plots and applications, such as in the <b>Jupyter/IPython notebook</b>.</li>
                <li><b>Client/server</b> architecture: Delegate computations to a remote server. (<i>in development</i>)</li>
                <li>Use a <b>cluster</b> to visualize and explore even larger datasets (10-100 billion). (<i>in development</i>)</li>
                <li>With a focus on astronomy and astrophysics, but widely applicable.</li>
                <li>Can visualize the whole <a href="#gaia"><b>Gaia catalogue</b></a> in one second.</li>
            </ul>
            </p>
        </div>
        <div class="col-md-4">
          <a href="image/nyc_taxi_full.png" data-toggle="lightbox" data-title="New York City Taxi pickup locations" data-footer=""  data-gallery="gallery-2" data-parent="" >
          <img class="featurette-image img-responsive center-block" src="image/nyc_taxi_square.png" alt="Generic placeholder image">
          </a>
        </div>
      </div>

      <hr class="featurette-divider">
      <a id="explore" name="explore"></a>
      <div class="row featurette">
        <div class="col-md-7 col-md-push-5">
          <h2 class="featurette-heading">How does it work?<!--<span class="text-muted">Make interactive selection.</span>--></h2>
            <p class="lead">
                vaex does this by
                <ul>
                    <li><b>Binning</b> or <b>aggregating</b> the data on a grid, using simple optimized algorithms</li>
                    <li><b>Columnar</b> storage of data avoids reading unneeded data and enables maximum performance of hard drives.</li>
                    <li><b>Memory mapped</b> files avoids unneeded reading, and copying of data.</li>
                </ul>
            </p>

            <!--<p class="lead">In the GUI, or the IPython/Jupyter notebook, a selection can be made in the plot, for instance a lasso selection. The selection is applied to all views of the data. In this example,
            a cluster is selected in the right view , and this cluster can be seen to correspond to a stream in the left view.
          </p>-->
        </div>
        <div class="col-md-5 col-md-pull-7">
          <a href="image/linked_views.png" data-toggle="lightbox" data-title="Helmi and de Zeeuw 2000" data-footer="Selection made on the right, also visible in the left"  data-gallery="global-gallery-2" data-parent="" >
            <img class="featurette-image img-responsive center-block" src="image/linked_views.png" alt="Generic placeholder image">
          </a>
        </div>
      </div>

      <hr class="featurette-divider">

      <div class="row featurette">
        <a id="flexible" name="flexible"></a>
        <div class="col-md-7">
          <h2 class="featurette-heading">What is vaex?</h2>
            <ul>
                <li>A program that</li>
                <ul>
                    <li>Visualizes 1d histograms, 2d density plots, averages quantities, and 3d volume rendering </li>
                    <li>Allows interactive <b>navigation</b> and <b>selection</b></li>
                    <li>Overlay <b>vector</b> and <b>tensor</b> quantities in 2 and 3d.</li>
                </ul>
                <li>A Python library/package for (data) scientists:</li>
                <ul>
                    <li>Is <b>pip</b> and <b>conda</b> installable.</li>
                    <li>Make <b>custom</b> plot and statistics.</li>
                    <li>Calculate <b>statistics</b> on a <b>N-dimensional grid</b> and visualize it.</li>
                    <li>Create interactive <b>Jupyter/IPython notebooks</b>.</li>
                    <li>Publication quality plots with <b>matplotlib</b>.</li>
                    <li>Interactive plots with <b>bqplot</b> or <b>bokeh</b>.</li>
                    <li>Combine the notebook with the graphical interface in one kernel</li>
                </ul>
            </ul>
          <!--<p class="lead">
            No tool will be able to give all options, we have to resort to programming. vaex can also be used as a library, which is especially usefull in the IPython/Jupyter notebook.
            Making custom plots, and even combining it with the gui, or doing selections programmatically are all possible.
          </p>
            <p class="lead">
            The vaex library for (data) scientists comes as a pip and conda installable Python package,
            for producing custom statistics, plots and interactive plots in the Jupyter/IPython Notebook.
            vaex supports matplotlib for publication quality plots, and bqplot and bokeh for interactivity.
            </p>-->
        </div>
        <div class="col-md-5">
          <img class="featurette-image img-responsive center-block" src="image/notebook.png"" alt="Generic placeholder image">
        </div>
      </div>

      <hr class="featurette-divider">
      <a id="demo" name="demo"></a>
      <div class="row featurette">
        <div class="col-md-7">
          <h2 class="featurette-heading">Live demo. <span class="text-muted">Yellow taxi pickup locations in New York City.</span></h2>
          <p class="lead">The demo on the right shows 140 million points, rendered real time. Zoom/pan and the plot get updated on the fly. </p>
        </div>

        <div class="col-md-5">
            <?php include 'nyt.html';?>
        </div>
      </div>



      <hr class="featurette-divider">

      <div class="row featurette">
        <a id="download" name="download"></a>
        <div class="col-md-5">
          <a class="download-vaex" href="#download"><i class="fa fa-download fa-big falink-black" aria-hidden="true"></i> </a>
        </div>
        <div class="col-md-7">
          <h2 class="featurette-heading">Download.</h2>

          <p class="lead">Desktop user? Download the standalone <a href="http://vaex.astro.rug.nl/program/vaex-1.0.0-beta.4-osx.zip">OSX</a> or
            <a href="http://vaex.astro.rug.nl/program/vaex-1.0.0-beta.4-linux.tar.gz">Linux</a> version. *

          </p>
          <p class="lead">For programming? Install the python package:
            <br/>
            <code>
              $ pip install --user --pre vaex
          </code>
          </p>
          <p class="lead">Or for anaconda users:
            <br/>
            <code>
              $ conda install -c conda-forge vaex
          </code>
          <p class="lead">Latest from git:
            <br/>
            <code>
              $ pip install git+https://github.com/maartenbreddels/vaex/
          </code>
          </p>
          <p class="lead">Or see <a href="http://vaex.astro.rug.nl/latest/installing.html">more detailed instructions</a>.</p>

          <span class="text-muted" style="font-size: 80%">*Not possible to combine with the IPython/Jupyter notebook</span>.

        </div>
      </div>



      <hr class="featurette-divider">


       <a id="example" name="example"></a>
      <div id="myCarousel-code" class="carousel slide" data-ride="carousel">
        <!-- Indicators -->
        <ol class="carousel-indicators">
          <li data-target="#myCarousel-code" data-slide-to="0" class="active"></li>
          <li data-target="#myCarousel-code" data-slide-to="1"></li>
        </ol>

        <!-- Wrapper for slides -->
        <div class="carousel-inner" role="listbox">
          <div class="item active">
            <div class="row featurette">
              <div class="col-md-8">
                <h2 class="featurette-heading">Python example.</h2>
      Run <code>python</code></codfe>, and paste:
<pre><code class="python">import vaex as vx
dataset = vx.datasets.helmi_de_zeeuw.fetch() # get a cup of coffee while this downloads
dataset.plot("Lz", "E", f="log1p", show=True)
</code></pre>
<a href="#" data-target="#myCarousel-code" data-slide-to="1">See next example, with a larger dataset</a>
              </div>
              <div class="col-md-4">
                <a href="image/Lz-E.png" data-toggle="lightbox" data-title="Energy vs angular momented" data-footer=""  data-gallery="gallery-2" data-parent="" >
                <img class="featurette-image img-responsive center-block" src="image/Lz-E.png" alt="Generic placeholder image">
                </a>
              </div>
            </div>

          </div>

          <div class="item">
            <div class="row featurette">
              <div class="col-md-8">
              <h2 class="featurette-heading">Notebook example.</h2>
      From the IPython/Jupter notebook, run
<pre><code class="python">import vaex as vx
dataset = vx.datasets.nyctaxi_yellow_201x.fetch() # get a cup of coffee while this downloads
dataset.plot_bq("pickup_longitude","pickup_latitude", f="log1p")
</code></pre>
The plot is interactive, meaning you can zoom in and out and the plot will be updated.
You will need about, ~15BG or free memory for a proper performance, or replace <code>nyc_taxi</code> by <code>nyc_taxi_2015</code> for a subset.
              </div>
              <div class="col-md-4">
                <a href="image/nyc_taxi_full.png" data-toggle="lightbox" data-title="New York City Taxi pickup locations" data-footer=""  data-gallery="gallery-2" data-parent="" >
                <img class="featurette-image img-responsive center-block" src="image/nyc_taxi_square.png" alt="Generic placeholder image">
                </a>
              </div>
            </div>
          </div>
        </div>

        <!-- Left and right controls -->
        <a class="left carousel-control" href="#myCarousel-code" role="button" data-slide="prev">
          <span class="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
          <span class="sr-only">Previous</span>
        </a>
        <a class="right carousel-control" href="#myCarousel-code" role="button" data-slide="next">
          <span class="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
          <span class="sr-only">Next</span>
        </a>
      </div>

      <hr class="featurette-divider">
      <div class="row featurette">
        <h2 class="featurette-heading">Demo movies.</h2>
        <div class="col-lg-4 col-md-4 col-sm-6 col-xs-12">
            <div class="hovereffect">
                <img class="img-responsive" src="image/aq_nytaxi_b.png" alt="">
                <div class="overlay">
                   <h2>Fast visualization</h2>
                   <a class="info" href="#">Coming soon</a>
                </div>
            </div>
        </div>
        <div class="col-lg-4 col-md-4 col-sm-6 col-xs-12">
            <div class="hovereffect">
                <img class="img-responsive" src="image/linked_views.png" alt="">
                <div class="overlay">
                   <h2>Selections and linked views</h2>
                   <a class="info" href="#">Coming soon</a>
                </div>
            </div>
        </div>
        <div class="col-lg-4 col-md-4 col-sm-6 col-xs-12">
            <div class="hovereffect">
                <img class="img-responsive" src="image/notebook.png" alt="">
                <div class="overlay">
                   <h2>Notebook integration</h2>
                   <a class="info" href="#">Coming soon</a>
                </div>
            </div>
        </div>
      </div>

      <hr class="featurette-divider">

       <a id="gaia" name="gaia"></a>
      <div class="row featurette">
        <h2 class="featurette-heading">Gaia data.</h2>


        <div class="col-md-12">
            <ul>
                <li>
                    See the <a href="http://www.cosmos.esa.int/web/gaia/home">Gaia Science Homepage for details</a>, and you may want to try the <a href="https://archives.esac.esa.int/gaia">Gaia Archive</a> for ADQL (SQL like) queries.
                </li>
            </ul>
            Data:
            <ul>
                <li>
                    Single hdf5 file, copy of <a href="http://cdn.gea.esac.esa.int/Gaia/gaia_source/fits/">full Gaia DR1 catalogue</a> in random row order: <a href="https://drive.google.com/file/d/0B8gjQokMGa4nUVc1bURQWVJNcnM/view?usp=sharing">direct download</a> (351G).
                </li>
		<li>
			random 10% of the catalogue, useful for on your laptop: <a href="https://drive.google.com/file/d/0B8gjQokMGa4nZWRvVXY5blQyaDg/view?usp=sharing">direct link</a> (35G).
                <li>
                    All rows, less columns (ra, dec, l, b,ra,dec, g magnitude, etc): <a href="https://drive.google.com/file/d/0B8gjQokMGa4nOF9YT2s5TE1aeW8/view?usp=sharing">direct download</a> (43G).
                </li>
                <li>
                    Single hdf5 file, copy of <a href="http://cdn.gea.esac.esa.int/Gaia/gaia_source/fits/">full TGAS catalogue</a>: <a href="data/tgas.hdf5">tgas-hdf5</a> (0.6G).
                </li>
            </ul>
        </div>

        <div class="col-md-12">
        <p>
          <p class="lead">Interactive demo showing 100 million points (10%) of the Gaia DR1 data, rendered real time. Zoom/pan and the plot gets updated on the fly. </p>
        </div>
        <div class="col-md-8">
       <a id="gaia" name="gaiademo"></a>
<?php include 'gaia-dr1.html';?>
        </div>

      </div>

      <hr class="featurette-divider">


        <div class="row featurette">
        <a id="about" name="about"></a>
        <div class="col-md-12">
          <h2 class="featurette-heading">Acknowledgements.</h2>
            Vaex is funded by:<br>
        <div class="col-lg-4 col-md-4 col-sm-6 col-xs-12">
            <a href="http://www.rug.nl/research/kapteyn/"><img width="300px" src="logos/rug_bigger.gif"></img</a>
        </div>
        <div class="col-lg-4 col-md-4 col-sm-6 col-xs-12">
            <a href="https://erc.europa.eu"><img width="300px" src="logos/erc2.jpg"></img</a>
        </div>
        <div class="col-lg-4 col-md-4 col-sm-6 col-xs-12">
            <a href="http://www.nova-astronomy.nl/"><img width="300px" src="logos/nova.jpg"></img</a>
        </div>
          <p class="lead">.</p>
        </div>
        <div class="col-md-5">
        </div>
      </div>

      <hr class="featurette-divider">
        <div class="row featurette">
        <a id="contact" name="concact"></a>
        <div class="col-md-12">
          <h2 class="featurette-heading">Requests/Issues/Contact.</h2>

          Vaex is open source, the source code and issues live on <a href="https://github.com/maartenbreddels/vaex"><i class="fa fa-github"></i>Github</a>. Please use <a href="https://github.com/maartenbreddels/vaex/issues">github to report issues</a>. Contributions are welcome using <a href="https://help.github.com/articles/about-pull-requests/">Pull Requests</a>.

<p>
            <ul>Contact the main author by email or on twitter:
                <li> <a href="mailto:breddels@astro.rug.nl"><i class="fa fa-envelope"></i> breddels@astro.rug.nl</a>
                </li>



                <li>
                        <a href="https://twitter.com/intent/tweet?screen_name=maartenbreddels" class="twitter-mention-button" data-show-count="false">Tweet to @maartenbreddels</a>
                        <a href="https://twitter.com/maartenbreddels" class="twitter-follow-button" data-show-count="false">Follow @maartenbreddels</a>
                </li>
                </ul>


          <p class="lead">.</p>
        </div>
        <div class="col-md-5">
        </div>
      </div>

      <!--<div class="row featurette">
        <a id="about" name="about"></a>
        <div class="col-md-7">
          <h2 class="featurette-heading">About.</h2>
          <p class="lead">.</p>
        </div>
        <div class="col-md-5">
        </div>
      </div>

      <hr class="featurette-divider">-->



      <hr class="featurette-divider">
      <!-- /END THE FEATURETTES -->

      <!-- FOOTER -->
      <footer>
        <p class="pull-right"><a href="#">Back to top</a></p>
        <!-- <p>&copy; 2015 Company, Inc. &middot; <a href="#">Privacy</a> &middot; <a href="#">Terms</a></p> -->
      </footer>

    </div><!-- /.container -->


    <!-- Bootstrap core JavaScript
    ================================================== -->
    <!-- Placed at the end of the document so the pages load faster -->
    <script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>
    <script src="https://npmcdn.com/jupyter-js-widgets@~1.1.2/dist/embed.js">
    <script src="//code.jquery.com/jquery.js"></script>
    <script src="js/jquery.client.js"></script>
    <script>window.jQuery || document.write('<script src="../../assets/js/vendor/jquery.min.js"><\/script>')</script>
    <script src="js/jquery.easing.min.js"></script>
    <script src="//maxcdn.bootstrapcdn.com/bootstrap/3.3.4/js/bootstrap.min.js"></script>
    <script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.5.0/highlight.min.js"></script>
    <script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.5.0/languages/python.min.js"></script>
    <script>hljs.initHighlightingOnLoad();</script>
    <!-- IE10 viewport hack for Surface/desktop Windows 8 bug -->
    <!--<script src="../../assets/js/ie10-viewport-bug-workaround.js"></script>-->

		<script src="//cdnjs.cloudflare.com/ajax/libs/ekko-lightbox/4.0.1/ekko-lightbox.js"></script>
		<script type="text/javascript">
			$(document).ready(function ($) {
			    if(($.client.os == "Linux") || ($.client.os == "Mac")) {
                    $('.download-vaex').on('click', function (e) {
                        e.preventDefault();
                        console.log($.client.os)
                        if($.client.os == "Mac")
                          window.location.href = "http://vaex.astro.rug.nl/program/vaex-1.0.0-beta.4-osx.zip"
                        if($.client.os == "Linux")
                          window.location.href = "http://vaex.astro.rug.nl/program/vaex-1.0.0-beta.4-linux.tar.gz"

                    });
                }
			   $('a.page-scroll').bind('click', function(event) {
                  var $anchor = $(this);
                  $('html, body').stop().animate({
                      scrollTop: $($anchor.attr('href')).offset().top
                  }, 800); // , 'easeInOutExpo'
                  event.preventDefault();
               });
        });

        $("#myModal").on("show.bs.modal", function(e) {
            var link = $(e.relatedTarget);
            $(this).find(".modal-body").load(link.attr("href"));
        });
		</script>
		<script type="text/javascript">
			$(document).ready(function ($) {

				// delegate calls to data-toggle="lightbox"
				$(document).delegate('*[data-toggle="lightbox"]:not([data-gallery="navigateTo"])', 'click', function(event) {
					event.preventDefault();
					return $(this).ekkoLightbox({
						onShown: function() {
							if (window.console) {
								return console.log('onShown event fired');
							}
						},
						onContentLoaded: function() {
							if (window.console) {
								return console.log('onContentLoaded event fired');
							}
						},
						onNavigate: function(direction, itemIndex) {
							if (window.console) {
								return console.log('Navigating '+direction+'. Current item: '+itemIndex);
							}
						}
					});
				});

				//Programatically call
				$('#open-image').click(function (e) {
					e.preventDefault();
					$(this).ekkoLightbox();
				});
				$('#open-youtube').click(function (e) {
					e.preventDefault();
					$(this).ekkoLightbox();
				});

				$(document).delegate('*[data-gallery="navigateTo"]', 'click', function(event) {
					event.preventDefault();
					return $(this).ekkoLightbox({
						onShown: function() {
							var lb = this;
							$(lb.modal_content).on('click', '.modal-footer a#jumpit', function(e) {
								e.preventDefault();
								lb.navigateTo(2);
							});
							$(lb.modal_content).on('click', '.modal-footer a#closeit', function(e) {
								e.preventDefault();
								lb.close();
							});
						}
					});
				});

			});
		</script>

<script type="text/javascript">
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-60052576-1', 'vaex.astro.rug.nl');
  ga('send', 'pageview');

</script>

  <!--<a href="https://github.com/maartenbreddels/vaex"><img style="position: absolute; top: 0; right: 0; border: 0; z-index: 1000;" src="https://camo.githubusercontent.com/e7bbb0521b397edbd5fe43e7f760759336b5e05f/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f677265656e5f3030373230302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_green_007200.png"></a>-->
  </body>
</html>
