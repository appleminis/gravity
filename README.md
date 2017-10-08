# gravity

## Overview
	Python OpenCL "gravitational" "fluid" simulation
	Not real gravitational potential : G/(D^2.3+eps) ( too much spreading speeds with G/(D+eps) )
	Pressure potential : -Density*PRESSURE
	Friction affect particles velocity sharing same discrete location

## Dependency

	OS
		QT4 OpenGL OpenCL
	PYTHON
		pyfft pyopencl pyopengl
## Python variables

	N -> number of particles
	D -> spatial discretization size

## CLkernel defines

	GRAVITY
	PRESSURE
	FRICTION

## Launch

	ipython gravity/galaxy.py

![Logo](/capture.png)

