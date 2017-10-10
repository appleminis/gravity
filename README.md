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

## Install

	sudo apt install ipython
	sudo apt install python-pip
	apt-cache search pyqt
	sudo apt-get install python-qt4
	sudo apt-get install libqt4-opengl
	sudo apt-get install python-qt4-gl
	sudo apt install ocl-icd-opencl-dev
	pip install numpy
	pip install pyopengl
	pip install pyfft
	sudo apt install python-cffi

	git clone https://github.com/inducer/pyopencl.git
	cd pyopencl/
	python configure.py --cl-enable-gl
	make -j 31
	sudo make install


![Logo](/capture.png)