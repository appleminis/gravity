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

import cv2

N=int(1*1e6)
D=int(1024)

ENC=1

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
    
    //calculate pressure in the "future"
    px+=vx;
    py+=vy;
    
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

        
        atomic_add(&d[ix+D*iy],(int)((1-dx)*(1-dy)*1000));
        atomic_add(&d[ix+D*iy+1],(int)((dx)*(1-dy)*1000));
        atomic_add(&d[ix+D*iy+D+1],(int)((dx)*(dy)*1000));
        atomic_add(&d[ix+D*iy+D],(int)((1-dx)*(dy)*1000));
        
        px-=vx;
        py-=vy;
        
        ix=px;
        iy=py;
    
        dx=px-ix;
        dy=py-iy;
        
        atomicAdd_g_f(&dv[p], vx);
        atomicAdd_g_f(&dv[p+1], vy);
        atomic_add(&dd[p/2],1);

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
#define GRAVITY 1.0f/10
#define Geps GRAVITY/3
#define D2 512
#define DS 1
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

    float dist = GRAVITY/(powr((cx*cx+cy*cy),1.3f)*DS+Geps);
  
    out[id2]=dist*d[id2];
    out[id2+1]=dist*d[id2+1];
}
"""

clkerpotential = """
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define PRESSURE 1.0f/500000
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void potential(
    __global const float *pgravity, __global const int *pdensity,  __global float *out)
{
    int p = get_global_id(0); 
    float d=pow(pdensity[p]/1000.0f,1.5f);
    out[p] = (-d*PRESSURE+pgravity[p*2])/1000;
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
#define FRICTION 0.075f
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

clkervisu = """
#define D 1024


inline float4 hsv2rgb(float H, float S, float V)
{
    float P, Q, T, fract;
    if (H>=360)
        H=0;
    else
        H /= 60;
    fract = H - floor(H);

    P = V*(1.0f - S);
    Q = V*(1.0f - S*fract);
    T = V*(1.0f - S*(1.0f - fract));

    if (0.0f <= H && H < 1.0f)
        return (float4)(V, T, P, 1.0f);
    else if (1.0f <= H && H < 2.0f)
        return (float4)(Q, V, P, 1.0f);
    else if (2.0f <= H && H < 3.0f)
        return (float4)(P, V, T, 1.0f);
    else if (3.0f <= H && H < 4.0f)
        return (float4)(P, Q, V, 1.0f);
    else if (4.0f <= H && H < 5.0f)
        return (float4)(T, P, V, 1.0f);
    else if (5.0f <= H && H < 6.0f)
        return (float4)(V, P, Q, 1.0f);
    else
        return (float4)(0.0f, 0.0f, 0.0f, 1.0f);

}
    

__kernel void visu(__write_only image2d_t out, __global int* dd, __global float* dv, __global uchar* dc)//, __global int* dd, __global float* dv)
{
   int x = get_global_id(0);
   int y = get_global_id(1); 
   int2 coords = (int2)(x,y);
   
   float density = dd[x+y*1024];
   if (density==0)
   {
       float4 rgb=(float4)0;
       write_imagef(out, coords, rgb);
       dc[(x+D*y)*3+0]=rgb.x*255;
       dc[(x+D*y)*3+1]=rgb.y*255;
       dc[(x+D*y)*3+2]=rgb.z*255;
   }
   else
   {
       float m = (float)sqrt(1.0f+density)/12;
       //float m = (float)log10(1.0f+density)/12;
       float2 v=(float2)(dv[2*(x+y*D)],dv[2*(x+y*D)+1])/dd[(x+y*D)];
       float dv = sqrt(v.x*v.x+v.y*v.y)*1000;
       float da = (atan2(v.y,v.x)/3.14159265359f+1.0f)/2;
       
       
       if (dv>1) dv=1;
       if (m>1) m=1;
       
       //float4 val = (float4)(m,m,m, 1.0f);
       
       float4 rgb=hsv2rgb(dv*360,sqrt(dv),m);
       //float4 rgb=hsv2rgb((1+cos((da*0+dv)*4*3.14259265359))*180,sqrt(dv+m*2),m);

       write_imagef(out, coords, (float4)(rgb.y, rgb.z, rgb.x, 1.0f));
       dc[(x+D*y)*3+0]=rgb.y*255;
       dc[(x+D*y)*3+1]=rgb.z*255;
       dc[(x+D*y)*3+2]=rgb.x*255;

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

    def set_data(self, data, datavit, vid):
        """Load 2D data as a Nx2 Numpy array.
        """
        self.data = data
        self.datavit = datavit
        self.count = data.shape[0]
        self.vidout = vid


    def initialize_buffers(self):
        """Initialize OpenGL and OpenCL buffers and interop objects,
        and compile the OpenCL kernel.
        """
        # empty OpenGL VBO
        self.glbuf = glvbo.VBO(data=np.zeros(self.data.shape),
                               usage=gl.GL_DYNAMIC_DRAW,
                               target=gl.GL_ARRAY_BUFFER)
        
        self.idtexgl = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.idtexgl)
        gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, D, D, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None);

                   
        self.glbuf.bind()
        # initialize the CL context
        self.ctx, self.queue = clinit()
        
        
        
        self.clglimage = cl.GLTexture(self.ctx, cl.mem_flags.READ_WRITE,gl.GL_TEXTURE_2D, 0, self.idtexgl, 2)  
        self.clbufim = cl.Buffer(self.ctx,
                            cl.mem_flags.READ_WRITE,
                            size=D*D*3)
        
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
        self.visu = cl.Program(self.ctx, clkervisu).build()

        self.plan = Plan((D,D), queue=self.queue)
        
        self.enctime=0
        self.rdbufint = np.zeros((D*D),np.int32)
        self.daint = np.zeros((D*D),np.int32)
        self.rdgl = np.zeros((D,D,3),np.uint8)
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
        self.queue.finish()

        

#        self.rdbufvit = np.zeros((D*D*2),np.float32)
#        cl.enqueue_copy(self.queue, self.rdbufvit, self.clbufdensityvit)
        self.enctime+=1
        if (np.mod(self.enctime,4)==0 and ENC==1):
            cl.enqueue_acquire_gl_objects(self.queue, [self.clglimage])
            self.visu.visu(self.queue, (D,D), None,self.clglimage,self.clbufdensityint,self.clbufdensityvit,self.clbufim)        
            cl.enqueue_release_gl_objects(self.queue, [self.clglimage])
            cl.enqueue_copy(self.queue, self.rdgl, self.clbufim)
            aff = np.reshape(self.rdgl,(D,D,3))
            self.vidout.write(aff[:,:,::-1].astype('uint8'))
            
            
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
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if (np.mod(self.enctime,4)==0 or ENC==0 or 1):

            if (1):
                
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.idtexgl)
                gl.glEnable(gl.GL_TEXTURE_2D)
                gl.glBegin(gl.GL_QUADS)
                gl.glTexCoord2f(0.0, 0.0)
                gl.glVertex2f(0, 0); 
                gl.glTexCoord2f(1.0, 0.0)
                gl.glVertex2f( 1.0, 0); 
                gl.glTexCoord2f(1.0, 1.0)
                gl.glVertex2f( 1.0, 1.0); 
                gl.glTexCoord2f(0.0, 1.0)
                gl.glVertex2f(0, 1.0);
                gl.glEnd()
                
            else:
                gl.glColor4d(0.5,0.7,0.8,0.04)
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
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
    if (ENC):
        vid=cv2.VideoWriter('./output.avi',fourcc, 20.0, (D,D))
    else:
        vid=None

    # define a Qt window with an OpenGL widget inside it
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            # generate random data points
            if (10):
                self.data = (np.random.rand(N,2)-.5)/2+.5
                theta = np.random.rand(N)*2*np.pi
                radius = pow(np.sqrt(np.random.rand(N)),1.5)/5
                self.data[:,0]=np.cos(theta)*radius+.5
                self.data[:,1]=np.sin(theta)*radius+.5
                self.datat=np.empty_like(self.data)
                self.datat[:,1]=self.data[:,0]-.5
                self.datat[:,0]=-self.data[:,1]+.5
                self.datavit = (np.random.rand(N,2)-.5)/100/15+self.datat/100/((0.125+radius[:,None]))/7
                self.data = np.array(self.data, dtype=np.float32)
                self.datavit = np.array(self.datavit, dtype=np.float32)
            else:
                self.data = (np.random.rand(N,2))
                self.data = (np.random.rand(N,2)-.5)/2+.5
                theta = np.random.rand(N)*2*np.pi
                radius = pow(np.sqrt(np.random.rand(N)),1.5)/2.5
                self.data[:,0]=np.cos(theta)*radius+.5
                self.data[:,1]=np.sin(theta)*radius+.5
                self.data = np.array(self.data, dtype=np.float32)
                radius = np.sqrt(np.sum((self.data-.5)**2,1))
                self.datat=np.empty_like(self.data)
                self.datat[:,1]=self.data[:,0]-.5
                self.datat[:,0]=-self.data[:,1]+.5
                self.datavit = (np.random.rand(N,2)-.5)/100/40+self.datat/100/((0.125+radius[:,None]))/17*((radius<0.3)[:,None])
                self.datavit = np.array(self.datavit, dtype=np.float32)


            # initialize the GL widget
            self.widget = GLPlotWidget()
            self.widget.set_data(self.data,self.datavit,vid)
            # put the window at the screen position (100, 100)
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)
            self.show()

    # create the Qt App and window
    app = QtGui.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    app.exec_()
    if (ENC):
        vid.release()
