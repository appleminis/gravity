
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:09:37 2017
@author: julien rodriguez-tao
"""
# PyQt4 imports
from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import QGLWidget
# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo
# PyOpenCL imports
import pyopencl as cl
from pyopencl.tools import get_gl_sharing_context_properties
from pyfft.cl import Plan
#import matplotlib.pyplot as plt


N=10000000
D=1024


clkernel = """
__kernel void clkernel(__global float2* clpos, __global float2* glpos)
{
    //get our index in the array
    unsigned int i = get_global_id(0);

    // copy the x coordinate from the CL buffer to the GL buffer
    glpos[i].x = clpos[i].x;
    glpos[i].y = clpos[i].y;

}
"""

clkeraddvit = """
#define D 512


__kernel void addvit(__global float2* clpos, __global float2* clvit)
{
    //get our index in the array
    unsigned int i = get_global_id(0);

    // copy the x coordinate from the CL buffer to the GL buffer
    clpos[i].x += clvit[i].x;
    clpos[i].y += clvit[i].y;
    
    float px=clpos[i].x*D;
    float py=clpos[i].y*D;
    
    if (px>D-1) clpos[i].x=1.0f/D;
    if (px<1) clpos[i].x=(D-1.0f)/D;
    if (py>D-1) clpos[i].y=1.0f/D;
    if (py<1) clpos[i].y=(D-1.0f)/D;

}
"""

clkersetzero= """
__kernel void setzero(__global float* d)
{
    //get our index in the array
    unsigned int i = get_global_id(0);

    // copy the x coordinate from the CL buffer to the GL buffer
    d[i]=0;

}
"""

clkerdensity = """
#define D 1024

inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
    union
    {
        unsigned int u32;
        float        f32;
    } next, expected, current;
    
    current.f32    = *addr;
    
    do
    {
        expected.f32 = current.f32;
        next.f32     = expected.f32 + val;
        current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
        expected.u32, next.u32);
    } 
    while( current.u32 != expected.u32 );
}

__kernel void density(__global float2* clpos, __global float2* clvit, __global int* d,  __global int* dd, __global float* dv)
{
    //get our index in the array
    unsigned int i = get_global_id(0);

    float px=clpos[i].x*D;
    float py=clpos[i].y*D;
    
    float vx=clvit[i].x;
    float vy=clvit[i].y;
    
    int ix=px;
    int iy=py;
    
    float dx=px-ix;
    float dy=py-iy;
    
    int p = 2*(ix+D*iy);
    float av,avx,avy;
    
    if (ix>=0 && iy>=0 && ix<D-1 && iy<D-1)
    {
        /*av=d[p/2];
        atomic_xchg(&d[p/2],av+1);*/

        
        atomic_add(&d[ix+D*iy],(int)((1-dx)*(1-dy)*1000*(99999*(i==-1)+1)));
        atomic_add(&d[ix+D*iy+1],(int)((dx)*(1-dy)*1000*(99999*(i==-1)+1)));
        atomic_add(&d[ix+D*iy+D+1],(int)((dx)*(dy)*1000*(99999*(i==-1)+1)));
        atomic_add(&d[ix+D*iy+D],(int)((1-dx)*(dy)*1000*(99999*(i==-1)+1)));
        
        /*atomic_add(&dv[2*(ix+D*iy)],(int)((1-dx)*(1-dy)*10000*vx));
        atomic_add(&dv[2*(ix+D*iy+1)],(int)((dx)*(1-dy)*10000*vx));
        atomic_add(&dv[2*(ix+D*iy+D+1)],(int)((dx)*(dy)*10000*vx));
        atomic_add(&dv[2*(ix+D*iy+D)],(int)((1-dx)*(dy)*10000*vx));

        atomic_add(&dv[2*(ix+D*iy)+1],(int)((1-dx)*(1-dy)*10000*vy));
        atomic_add(&dv[2*(ix+D*iy+1)+1],(int)((dx)*(1-dy)*10000*vy));
        atomic_add(&dv[2*(ix+D*iy+1025)+1],(int)((dx)*(dy)*10000*vy));
        atomic_add(&dv[2*(ix+D*iy+D)+1],(int)((1-dx)*(dy)*10000*vy));*/
        
        atomicAdd_g_f(&dv[p], vx);
        atomicAdd_g_f(&dv[p+1], vy);
        atomic_add(&dd[p/2],1);
        
        /*avx = dv[p];
        atomic_xchg(&dv[p],avx+vx);
        avy = dv[p+1];
        atomic_xchg(&dv[p+1],avy+vy);*/

    }
}


"""

clkertocomplex = """
__kernel void tocomplex(
    __global const int *d, __global float *out)
{
    int px = get_global_id(0);  
  
    out[2*px]=(float)d[px]/1000;
    out[2*px+1]=0.0f;
    
}
"""

clkergravity = """
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define GRAVITY 1.0f/40
#define Geps GRAVITY/6
#define D2 512
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void gravity(
    __global const float *d, __global float *out)
{
    int px = get_global_id(0);
    int py = get_global_id(1);  
    
    int id2 = 2*(px+get_global_size(0)*py);
    
            
    px+=(px<D2 ? D2:-D2);
    py+=(py<D2 ? D2:-D2);
    
    float cx=px-D2;
    float cy=py-D2;

    float dist = (GRAVITY)/(pow(cx*cx+cy*cy,1.15f)+Geps);
  
    out[id2]=dist*d[id2];
    out[id2+1]=dist*d[id2+1];
}
"""

clkerpotential = """
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define PRESSURE 1.0f/200000000
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void potential(
    __global const float *pgravity, __global const int *pdensity,  __global float *out)
{
    int p = get_global_id(0);  
    out[p] = (-pdensity[p]*PRESSURE+pgravity[p*2])/1000;
}
"""

clkergrad = """
__kernel void grad(
    __global const float *d, __global float2 *out)
{
    int px = get_global_id(0);
    int py = get_global_id(1);  
    
    int id = (px+get_global_size(0)*py);
    
    if (px>0 && px<get_global_size(0)-1)
        out[id].x=d[id+1]-d[id-1];
    else
        out[id].x=0;
    
    if (py>0 && py<get_global_size(1)-1)
        out[id].y=d[id+get_global_size(0)]-d[id-get_global_size(0)];
    else
        out[id].y=0;
}
"""

clkeracceleration = """
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define FRICTION 0.005f
#define D 1024
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void acceleration(
      __global float2* clvit, __global float2* clpos, __global float2 *acc, __global int* d, __global int* dd, __global float* dv)
{
    int p = get_global_id(0);
    
    float2 pp = clpos[p]*D;
    float ntm;
    
    int ix=pp.x;
    int iy=pp.y;
    
    float dx=pp.x-ix;
    float dy=pp.y-iy;
    
    if (pp.x>=0 && pp.y>=0 && pp.x<D-1 && pp.y<D-1)
    {

        //float dp = (float)d[ix+iy*1024]*(1-dx)*(1-dy)+(float)d[ix+iy*1024+1]*(dx)*(1-dy)+(float)d[ix+iy*1024+1024]*(1-dx)*(dy)+(float)d[ix+iy*1024+D+1]*(dx)*(dy);
        //float dvx = (float)dv[2*(ix+iy*1024)]*(1-dx)*(1-dy)+(float)dv[2*(ix+iy*1024+1)]*(dx)*(1-dy)+(float)dv[2*(ix+iy*1024+1024)]*(1-dx)*(dy)+(float)dv[2*(ix+iy*1024+1025)]*(dx)*(dy);
        //float dvy = (float)dv[2*(ix+iy*1024)+1]*(1-dx)*(1-dy)+(float)dv[2*(ix+iy*1024+1)+1]*(dx)*(1-dy)+(float)dv[2*(ix+iy*1024+1024)+1]*(1-dx)*(dy)+(float)dv[2*(ix+iy*1024+1025)+1]*(dx)*(dy);

        clvit[p] = clvit[p]*(1-FRICTION)+(float2)(dv[2*(ix+iy*D)],dv[2*(ix+iy*D)+1])/dd[(ix+iy*D)]*FRICTION;
        clvit[p]+=(acc[ix+iy*D]*(1-dx)*(1-dy)+acc[ix+iy*D+1]*(dx)*(1-dy)+acc[ix+iy*D+D]*(1-dx)*(dy)+acc[ix+iy*D+D+1]*(dx)*(dy))/(1+99999*(p==-1));
    }
        
}
"""

def clinit():
    """Initialize OpenCL with GL-CL interop.
    """
    plats = cl.get_platforms()
    # handling OSX
    if sys.platform == "darwin":
        ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                             devices=[])
    else:
        ctx = cl.Context(properties=[
                            (cl.context_properties.PLATFORM, plats[0])]
                            + get_gl_sharing_context_properties())
    queue = cl.CommandQueue(ctx)
    return ctx, queue

class GLPlotWidget(QGLWidget):
    # default window size
    width, height = D,D

    def set_data(self, data, datavit):
        """Load 2D data as a Nx2 Numpy array.
        """
        self.data = data
        self.datavit = datavit

        self.count = data.shape[0]

    def initialize_buffers(self):
        """Initialize OpenGL and OpenCL buffers and interop objects,
        and compile the OpenCL kernel.
        """
        # empty OpenGL VBO
        self.glbuf = glvbo.VBO(data=np.zeros(self.data.shape),
                               usage=gl.GL_DYNAMIC_DRAW,
                               target=gl.GL_ARRAY_BUFFER)
        self.glbuf.bind()
        # initialize the CL context
        self.ctx, self.queue = clinit()
        # create a pure read-only OpenCL buffer
        self.clbuf = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=self.data)
                            
        self.clbufvit = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                            hostbuf=self.datavit)
                            
        self.clbufdensity = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=D*D*4)
                            
        self.clbufdensityint = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=D*D*4)
                            
        self.clbufdensityvit = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=D*D*8)
                            
        self.clbufdensityc = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=D*D*8)
                            
        self.clbuffft = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=D*D*8)
                            
        self.clbufifft = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=D*D*8)
                            
        self.clbufpotential = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=D*D*4)
                            
        self.clbufgrad = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=D*D*8)
                            
        # create an interop object to access to GL VBO from OpenCL
        self.glclbuf = cl.GLBuffer(self.ctx, cl.mem_flags.READ_WRITE,
                            int(self.glbuf.buffers[0]))
        # build the OpenCL program
        self.program = cl.Program(self.ctx, clkernel).build()
        self.addvit = cl.Program(self.ctx, clkeraddvit).build()
        self.setzero = cl.Program(self.ctx, clkersetzero).build()
        self.density = cl.Program(self.ctx, clkerdensity).build()
        self.tocomplex = cl.Program(self.ctx, clkertocomplex).build()
        self.gravity = cl.Program(self.ctx, clkergravity).build()
        self.potential = cl.Program(self.ctx, clkerpotential).build()
        self.grad = cl.Program(self.ctx, clkergrad).build()
        self.acceleration = cl.Program(self.ctx, clkeracceleration).build()



        self.plan = Plan((D,D), queue=self.queue)



        # release the PyOpenCL queue
        self.queue.finish()

    def execute(self):
        """Execute the OpenCL kernel.
        """
        self.setzero.setzero(self.queue, (D*D,), None, self.clbufdensity)  
        self.setzero.setzero(self.queue, (D*D,), None, self.clbufdensityint)        
        self.setzero.setzero(self.queue, (D*D*2,), None, self.clbufdensityvit)        

        self.density.density(self.queue, (self.count,), None, self.clbuf,self.clbufvit, self.clbufdensity, self.clbufdensityint, self.clbufdensityvit)        
        self.tocomplex.tocomplex(self.queue, (D*D,), None, self.clbufdensity, self.clbufdensityc)        
        self.plan.execute(self.clbufdensityc,self.clbuffft,wait_for_finish=True)
        self.gravity.gravity(self.queue, (D,D), None, self.clbuffft,self.clbufifft)
        self.plan.execute(self.clbufifft,self.clbuffft,wait_for_finish=True,inverse=True)
        self.potential.potential(self.queue, (D*D,), None, self.clbuffft, self.clbufdensity, self.clbufpotential)
        self.grad.grad(self.queue, (D,D), None, self.clbufpotential,self.clbufgrad)
        self.acceleration.acceleration(self.queue, (self.count,), None , self.clbufvit, self.clbuf, self.clbufgrad,self.clbufdensity,self.clbufdensityint,self.clbufdensityvit)
        self.addvit.addvit(self.queue, (self.count,), None, self.clbuf,self.clbufvit)        

        
        # get secure access to GL-CL interop objects
        cl.enqueue_acquire_gl_objects(self.queue, [self.glclbuf])
        # arguments to the OpenCL kernel
        kernelargs = (self.clbuf,
                      self.glclbuf)
        # execute the kernel
        self.program.clkernel(self.queue, (self.count,), None, *kernelargs)
        # release access to the GL-CL interop objects
        cl.enqueue_release_gl_objects(self.queue, [self.glclbuf])
        self.queue.finish()

#        self.rdbufvit = np.zeros((D*D*2),np.float32)
#        cl.enqueue_copy(self.queue, self.rdbufvit, self.clbufdensityvit)
#        self.rdbufint = np.zeros((D*D),np.int32)
#        cl.enqueue_copy(self.queue, self.rdbufint, self.clbufdensityint)
#        self.rvx=np.reshape(self.rdbufvit[::2]/self.rdbufint,(D,D))
#        self.rvy=np.reshape(self.rdbufvit[1::2]/self.rdbufint,(D,D))


        #plt.imshow(np.reshape(self.rdbufvit,(D,D*2)))        
        

    def update_buffer(self):
        """Update the GL buffer from the CL buffer
        """
        # execute the kernel before rendering
        self.execute()
        gl.glFlush()

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        # initialize OpenCL first
        self.initialize_buffers()
        # set background color
        gl.glClearColor(0,0,0,.5)
        # update the GL buffer from the CL buffer
        self.update_buffer()

    def paintGL(self):
        """Paint the scene.
        """

        self.update_buffer()   
        
        # clear the GL scene
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # set yellow color for subsequent drawing rendering calls
        gl.glColor4f(0.5,0.7,0.8,0.01)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendEquationSeparate( gl.GL_FUNC_ADD,  gl.GL_FUNC_ADD);
        gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA, gl.GL_ONE,    gl.GL_ONE, gl.GL_ZERO);
        # bind the VBO
        self.glbuf.bind()
        # tell OpenGL that the VBO contains an array of vertices
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        # these vertices contain 2 simple precision coordinates
        gl.glVertexPointer(2, gl.GL_FLOAT, 0, self.glbuf)
        # draw "count" points from the VBO
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        
        self.update()
                
    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, width, height)
        # set orthographic projection (2D only)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # the window corner OpenGL coordinates are (-+1, -+1)
        gl.glOrtho(0, 1, 0, 1, 0, 1)

window = None

if __name__ == '__main__':
    import sys
    import numpy as np

    # define a Qt window with an OpenGL widget inside it
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            # generate random data points
            self.data = (np.random.rand(N,2)-.5)/2+.5
            theta = np.random.rand(N)*2*np.pi
            radius = np.sqrt(np.random.rand(N))/5
            self.data[:,0]=np.cos(theta)*radius+.5
            self.data[:,1]=np.sin(theta)*radius+.5
            self.datat=np.empty_like(self.data)
            self.datat[:,1]=self.data[:,0]-.5
            self.datat[:,0]=-self.data[:,1]+.5
            self.datavit = (np.random.rand(N,2)-.5)/100/4+self.datat/100/1.2
            self.data = np.array(self.data, dtype=np.float32)
            self.datavit = np.array(self.datavit, dtype=np.float32)
            self.data[0,:]=0.5
            self.datavit[0,:]=0
            # initialize the GL widget
            self.widget = GLPlotWidget()
            self.widget.set_data(self.data,self.datavit)
            # put the window at the screen position (100, 100)
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)
            self.show()

    # create the Qt App and window
    app = QtGui.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    app.exec_()
    
#    import matplotlib.pyplot as plt
#    dns = np.zeros((D,D),np.complex64)
#    cl.enqueue_copy(window.widget.queue, dns, window.widget.clbuffft)
#    plt.imshow(np.abs(dns))