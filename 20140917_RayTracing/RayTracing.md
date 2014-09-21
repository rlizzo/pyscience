Title: From Ray Casting to Ray Tracing with Python and VTK
Author: Adamos Kyriakou
Date: Friday September 17th, 2014
Tags: Python, IPython Notebook, VTK, STL, Ray Tracing, Ray Casting
Categories: Image Processing, Visualization, VTK

In this post I will show how to use VTK to trace rays emanating from a source mesh, intersecting with another target mesh, and then show how to cast subsequent rays bouncing off the target. This will include calculating of normal vectors at the mesh cells, vector visualization through glyphs, ???

# Introduction

## Background
In my previous post???, I used Python and VTK to show you how to perform ray-casting, i.e., intersection tests between arbitrary lines/rays and a mesh and extraction of the intersection point coordinates through the [`vtkOBBTree`](http://www.vtk.org/doc/release/5.2/html/a00908.html) class.

## Summary

# Ray-Tracing with Python & VTK

## Imports

## Helper-functions

## Options

## 'Environment' Creation





## What about ray-tracing?
If you want to chase that ray and pursue the noble sport of ray-tracing there's tons of material out there so Google's thy friend. Now I never had to do that myself so I didn't bother implementing it but if I had to here's **how I'd go about it** (using VTK):

- I would assign materials to all objects in my scene, each material featuring its own attenuation, and the physical properties that would define the refractive indices. Now these properties would differ depending on what physics you're dealing with, e.g., optics, acoustics, etc, but the concepts are the same.
- I would cast a ray with a given energy/intensity and use the `vtkOBBTree.IntersectWithLine` method to test for intersections with every object in the scene. The one difference from the calls we saw before is that we would want to know the ids of the triangular mesh cells where those intersections happened (we'll see why later). We'd do that as follows:

```
idsCells = vtk.vtkIdList()
obbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection, idsCells)
```
 
- For every object that did exhibit intersection points with that ray I would get the first such point and get its distance from `pSource` using `vtkMath.Distance2BetweenPoints` immediately 'forgetting' all other objects and points.
- In order to calculate the directions of the reflected and refracted rays we would then need to calculate the normal vector for the first intersected cell in the `mesh` of that solid as such:

```
normals = vtk.vtkPolyDataNormals()
normals.SetInput(mesh)
normals.ComputeCellNormalsOn()
normals.ComputePointNormalsOff()
normals.Update()

arrayNormals = n.GetOutput().GetCellData().GetNormals()
vectorNormal = arrayNormals.GetTuple(idsCells.GetId(0))
```

> To be honest, VTK won't understand, or at least not mind, that the coordinates we're passing reside in a `numpy.ndarray`. The only reason I'm bothering converting the `numpy.ndarray` object back to lists are for the sake of consistency. However, you should be careful with the funky objects you

[http://asalga.wordpress.com/2012/09/23/understanding-vector-reflection-visually/](http://asalga.wordpress.com/2012/09/23/understanding-vector-reflection-visually/)

[http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf](http://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf)

[https://answers.yahoo.com/question/index?qid=20120512144935AA9gJxY](https://answers.yahoo.com/question/index?qid=20120512144935AA9gJxY)