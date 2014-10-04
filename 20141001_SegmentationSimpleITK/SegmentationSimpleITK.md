Title: Image Segmentation with Python and SimpleITK
Author: Adamos Kyriakou
Date: Friday October 4th, 2014
Tags: Python, IPython Notebook, DICOM, VTK, ITK, SimpleITK, Medical Image Processing, Image Segmentation
Categories: Image Processing, Visualization, VTK

In this post I will show how to use SimpleITK, an abstraction layer over the ITK library, to segment/label the white and gray matter from an MRI medical image dataset. This will include loading a DICOM series, image smoothing, region-growing and connectivity image filters, binary hole filling, as well as visualization tricks.

# Introduction
I think that by this point you've had enough of [VTK](http://www.vtk.org/) and its obscure, bordering on the occult, inner-workings. I admit that the topics I covered so far were not that much about visualization but mostly about 'secondary' functionality of VTK. However, information and examples on the former can be found with relative ease and I've posted a few such links in this [early post about IPython & VTK](http://pyscience.wordpress.com/2014/09/03/ipython-notebook-vtk/).

Today I'll be branching off and talking about its cousin, the [Insight Segmentation and Registration Toolkit (ITK)](http://www.itk.org/), a library created by the same folk as [VTK](http://www.vtk.org/), i.e., [Kitware](http://www.kitware.com/), but which focuses on image processing.

## Background

### Insight Toolkit (ITK)
[ITK](http://www.itk.org/) includes a whole bunch of goodies including routines for the segmentation, registration, and interpolation of multi-dimensional image data.

Just like [VTK](http://www.vtk.org/), [ITK](http://www.itk.org/) exhibits the same mind-boggling design paradigms and implementation, near-inexistent documentation (apart from [this one book](http://www.itk.org/ITK/help/book.html) and [some little tidbits](http://www.itk.org/Wiki/ITK/Documentation) like a couple presentations and webinars.

Nonetheless, just like [VTK](http://www.vtk.org/), [ITK](http://www.itk.org/) offers some amazing functionality one can't just overlook. Its no coincidence ITK is being heavily employed in image processing software like [MeVisLab](http://www.mevislab.de/) and [3DSlicer](http://www.slicer.org/). It just works!

### SimpleITK
However, today we won't be dealing with ITK, but [SimpleITK](http://www.simpleitk.org/) instead! [SimpleITK](http://www.simpleitk.org/) is, as the name implies, a simplified layer/wrapper build on top of ITK, exposing the vast majority of [ITK](http://www.itk.org/) functionality through bindings in a variety of languages (we only care about Python here anyway), greatly simplifying its usage.

While the usage of [ITK](http://www.itk.org/) would require incessant usage of templates and result in code like this:

```
// Setup image types.typedef float InputPixelType;
typedef float OutputPixelType;
typedef itk::Image<InputPixelType, 2> InputImageType;
typedef itk::Image<OutputPixelType, 2> OutputImageType;
// Filter type
typedef itk::DiscreteGaussianImageFilter<InputImageType, OutputImageType> FilterType;

// Create a filter
FilterType::Pointer filter = FilterType::New();
// Create the pipelinefilter−>SetInput(reader−>GetOutput());filter−>SetVariance(1.0);filter−>SetMaximumKernelWidth(5);filter−>Update();OutputImageType::Pointer blurred = filter−>GetOutput();```

[SimpleITK](http://www.simpleitk.org/) hides all that spaghetti-code and, by providing Python-bindings, yields something like this:

```
import SimpleITK
imgInput = SimpleITK.ReadImage(filename)imgOutput = SimpleITK.DiscreteGaussianFilter(imgInput, 1.0, 5)
```

However, the power of [SimpleITK](http://www.simpleitk.org/), doesn't end in allowing for succinct ITK calls. Some very notable features are:
- Super-simple IO of multi-dimensional image data supporting most image file formats (including DICOM through GDCM, more on that later)
- A fantastic `SimpleITK.Image` class which handles all image data and which includes overloads all basic arithmetic (`+` `-` `*` `/` `//` `**`) and binary (`&` `|` `^` `~`) operators. These operators just wrap the corresponding ITK filter and operate on a pixel-by-pixel basis thus allowing you to work directly on the image data without long function calls and filter object initialization.
- Slicing capability akin to that seen in `numpy.ndarray` objects allowing you to extract parts of the image, crop it, tile it, flip it etc etc. While slicing in [SimpleITK](http://www.simpleitk.org/) is not as powerful as NumPy its still pretty damn impressive!
- Built-in support for two-way conversion between `sitk.Image` object and `numpy.ndarray` (there's a catch though, more on that later as well)
- etc etc

If you want to read up on [SimpleITK](http://www.simpleitk.org/) then check out the following resources:
- [SimpleITK Tutorial for MICCAI 2011](https://github.com/SimpleITK/SimpleITK-MICCAI-2011-Tutorial): A GitHub repo containing the material for a [SimpleITK](http://www.simpleitk.org/) tutorial presented at MICCAI 2011. Its quite a good introduction and you can find the presentation as a .pdf [here](https://github.com/SimpleITK/SimpleITK-MICCAI-2011-Tutorial/blob/master/Presentation/SimpleITK-MICCAI-2011.pdf).
- [SimpleITK Notebooks](http://simpleitk.github.io/SimpleITK-Notebooks/): A collection of IPython Notebooks showcasing different features of [SimpleITK](http://www.simpleitk.org/). You might also want to check the [GitHub repo](https://github.com/SimpleITK/SimpleITK-Notebooks) of the above page which contains the demo-data and some extra notebooks. However, keep in mind that some of the code, particularly that dealing with fancy IPython widget functionality, won't work straight out. Nonetheless, its the best resource out there. Another nice IPython Notebook entitled 'SimpleITK Image Filtering Tutorial' can be found [here](http://nbviewer.ipython.org/github/reproducible-research/scipy-tutorial-2014/blob/master/notebooks/01-SimpleITK-Filtering.ipynb).
- [SimpleITK Examples](http://itk.org/gitweb?p=SimpleITK.git;a=tree;f=Examples;hb=HEAD): A small number of basic examples in C++ and Python which showcases some of the [SimpleITK](http://www.simpleitk.org/) functionality. The number of demonstrated classes, however, is rather small.

## Installation
For better or for worse, [SimpleITK](http://www.simpleitk.org/) isn't written in pure Python but rather C++ which translates to you needing a compiled version of the library. Unfortunately, [SimpleITK](http://www.simpleitk.org/)  doesn't come pre-compiled with any of the major alternative Python distros I know so installing it is a lil' more hairy than normal.

### Vanilla Python
If you're using a vanilla Python intepreter, i.e., a Python distro downloaded straight from [python.org](https://www.python.org/), or the 'system Python' that comes pre-installed with most Linux and OSX distros, then you're in luck! You can simply install the [SimpleITK package](https://pypi.python.org/pypi/SimpleITK/0.8.0) hosted on PyPI through `pip` as such:

```
pip install SimpleITK
``` 

or by using `easy_install` and one of the [Python eggs (`.egg`)](http://peak.telecommunity.com/DevCenter/PythonEggs) or [Python wheels (`.whl`)](http://pythonwheels.com/) provided at the [SimpleITK download page](http://www.simpleitk.org/SimpleITK/resources/software.html) as such:

```
easy_install <filename>
```

### Anaconda Python
Unfortunately, here's where the trouble starts. Alternative Python distros like the [Anaconda Python](https://store.continuum.io/cshop/anaconda/), [Enthought Python](https://www.enthought.com/products/epd/), [Enthought Canopy](https://www.enthought.com/products/canopy/), or even the OSX [MacPorts](https://www.macports.org/) and [Homebrew](http://brew.sh/) Python, have their own version of the Python interpreters and dynamic libraries. 

However, the aforementioned [SimpleITK wheels and eggs](http://www.simpleitk.org/SimpleITK/resources/software.html) are all compiled and linked against vanilla Python interpreters, and should you make the noob mistake of installing on of those in a non-vanilla environment, then upon importing that package you'll most likely get the following infamous error:

```
Fatal Python error: PyThreadState_Get: no current thread
Abort trap: 6
```

This issue, however, isn't exclusive to [SimpleITK](http://www.simpleitk.org/). Any package containing Python-bound C/C++ code compiled against a vanilla Python will most likely result in the above error if used in a different interpreter. Solution? To my knowledge you have one of three options:

1. Compile the code yourself against the Python interpreter and dynamic library you're using. 
	
	That of course includes checking out the code, ensuring you already have (or even worse compile from source) whatever dependencies that package has, and build the whole thing yourself. If you're lucky it'll will just take a couple hours of tinkering. If not you can spend days trying to get the thing to compile without errors, repeating the same bloody procedure over and over and harassing people online for answers.

	In the case of [SimpleITK](http://www.simpleitk.org/) there are instructions on how to do so under the 'Building using SuperBuild' under their [Getting Started page](http://www.itk.org/Wiki/SimpleITK/GettingStarted). However, a process like this typically assumes a working knowledge of [Git](http://git-scm.com/), [CMake](http://www.cmake.org/), and the structure of your non-vanilla Python distribution.

2. Wait patiently till the devs of your non-vanilla Python distro build the package for you and give you a convenient way of installing it. 

	Companies like [Continuum Analytics](http://www.continuum.io/) (creators of Anaconda Python) and [Enthought](https://www.enthought.com/) often do the work for you. That's exactly why you have packages like VTK all built and ready with those distros. However, in the case of less-popular packages, [SimpleITK](http://www.simpleitk.org/) being one, you might be waiting for some time till enough people request it and the devs take time to do so.
	
3. Depend on the kindness of strangers. Often enough, other users of non-vanilla Python distros will do the work described in (1) and, should they feel like it, distribute the built package for other users of the same distro. 

This last one is exactly the case today. I compiled the latest release (v0.8.0) of [SimpleITK](http://www.simpleitk.org/) against an x64 [Anaconda Python](https://store.continuum.io/cshop/anaconda/) with Python 2.7 under Windows 8.1 and Linux Mint 17 and I'm gonna give you the .egg files you need to install the package. You can get this [Linux egg under here](https://bitbucket.org/somada141/pyscience/raw/master/20141001_SegmentationSimpleITK/Material/SimpleITK-0.8.0.post47-py2.7-linux-x86_64.egg) and the [Windows egg under here](https://bitbucket.org/somada141/pyscience/raw/master/20141001_SegmentationSimpleITK/Material/SimpleITK-0.8.0.post47-py2.7-win-amd64.egg) (both hosted on the [blog's BitBucket repo](https://bitbucket.org/somada141/pyscience)).

> Here I should note that the above, i.e., people distributing their own builds to save other people the trouble of doing so themselves, is not that uncommon. Its actually the primary reason behind the creation of [Binstar](https://binstar.org), a package distribution system by the creators of Anaconda Python, principally targeting that distro, meant to allow users to redistribute binary builds of packages and permitting them to be installed through `conda`. There you will find many custom-built packages such as OpenCV, PETSc, etc but I'll get back to [Binstar](https://binstar.org) at a later post. However, I didn't find the time to package [SimpleITK](http://www.simpleitk.org/) on Binstar hence .egg files it is for now :).

If you're using Anaconda Python the you simply need to download the appropriate `.egg` and install it through `easy_install <egg filename>`. If you're using an Anaconda environment, e.g. named `py27` environment as instructed in [this past post](pyscience.wordpress.com/2014/09/01/anaconda-the-creme-de-la-creme-of-python-distros-3/), then you need to activate that environment prior to installing the package through `activate py27` (Windows) or `source activate py27` on Linux.

You might be wondering where the Mac OSX `.egg` is. To my dismay I did not manage to build [SimpleITK](http://www.simpleitk.org/) against [Anaconda Python](https://store.continuum.io/cshop/anaconda/) on OSX due to a known CMake issue (which results in the build being linked against the pre-installed system python despite the CMake configuration). I've [extensively pestered](https://github.com/conda/conda-recipes/issues/178#issuecomment-55947595) Brad Lowekamp, the current maintainer of [SimpleITK](http://www.simpleitk.org/), over this but we haven't figured it out so far. Should I manage to build it at some point I will update this post.

## Summary 
The purpose of today's post was to introduce you to SimpleITK, show you how to install it, and give you a taste of its image-processing prowess. 

Personally, when I started this blog I intended to only address challenging yet IMHO interesting topics that I once found hard to tackle, and powerful tools that either suffered from poor/insufficient documentation or were not as prominent in the community as they deserve to be.

[SimpleITK](http://www.simpleitk.org/) falls under the latter category. I find the package to be as easy and Pythonic as a Python-bound C++ package can be. In addition, there's a whole treasure-trove of functionality one just can't find elsewhere. However, I find the existing documentation lacking and perhaps its due to the fact that people don't know of its existence.

Hence, today I'll do a little demonstration of some of [SimpleITK's](http://www.simpleitk.org/) functionality, and use it to semi-automatically segment the brain-matter (white and gray) off an MRI dataset of my own head (of which I first spoke in [this past post about DICOM in Python](http://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/)). You can download that dataset [here](https://bitbucket.org/somada141/pyscience/raw/master/20141001_SegmentationSimpleITK/Material/MyHead.zip) and you should extract its contents alongside [today's notebook](http://nbviewer.ipython.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141001_SegmentationSimpleITK/Material/SegmentationSimpleITK.ipynb).

The process will include loading the series of DICOM files into a single `SimpleITK.Image` object, smoothing that image to reduce noise, segmenting the tissues using region-growing techniques, filling holes in the resulting tissue-labels, while I'll also show you a few visualization tricks that come with SimpleITK.

However, keep in mind that the presented functionality is the mere tip of a massive iceberg and that [SimpleITK](http://www.simpleitk.org/) offers a lot more. To prove that point, I repeated the process in [today's notebook](http://nbviewer.ipython.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141001_SegmentationSimpleITK/Material/SegmentationSimpleITK.ipynb) in an '[alternative notebook](http://nbviewer.ipython.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141001_SegmentationSimpleITK/Material/SegmentationSimpleITK_AltFilters.ipynb)' where I used different techniques to achieve similar results (feel free to take a look cause I won't be going over this '[alternative notebook](http://nbviewer.ipython.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141001_SegmentationSimpleITK/Material/SegmentationSimpleITK_AltFilters.ipynb)' today). In addition, and as I mentioned in the intro, SimpleITK comes with a lot of classes tailored to image registration, interpolation, etc etc. I may demonstrate things like that in later posts.

# Image Segmentation

## Imports

## Helper-Functions

## DICOM Input

## Smoothing

## Segmentation: Region-Growing