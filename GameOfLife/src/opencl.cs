using Cloo;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using System;
using System.IO;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Template
{
    // OpenCLImage
    // Encapsulates an integer or float OpenGL texture, which can be bound to OpenCL as an
    // ComputeImage2D. Writing to this image allows OpenCL to produce output directly to
    // the on-device OpenGL texture, elliminating transfer of pixels to and from the host.
    public class OpenCLImage<T> where T : struct
    {
        public int OpenGLTextureID;
        public ComputeImage2D texBuffer;
        T [] texData;
        public OpenCLImage( OpenCLProgram ocl, int width, int height )
        {
            texData = new T[width * height * 4];
            OpenGLTextureID = GL.GenTexture();
            GL.BindTexture( TextureTarget.Texture2D, OpenGLTextureID );
            GL.TexParameter( TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest );
            GL.TexParameter( TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest );
            Type itemType = typeof( T );
            if (itemType == typeof( int ))
            {
                // create an integer texture (RGBA8, 32bit per pixel)
                GL.TexImage2D( TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba8, width, height, 0, OpenTK.Graphics.OpenGL.PixelFormat.Rgb, PixelType.Int, texData );
            }
            else if (itemType == typeof( float ))
            {
                // create a floating point texture (RGBA32, 128bit per pixel)
                GL.TexImage2D( TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba32f, width, height, 0, OpenTK.Graphics.OpenGL.PixelFormat.Rgb, PixelType.Float, texData );
            }
            else
            {
                ocl.FatalError( "Unsupported OpenCLImage format" );
            }
            texBuffer = ComputeImage2D.CreateFromGLTexture2D( ocl.context, ComputeMemoryFlags.WriteOnly, (int)TextureTarget.Texture2D, 0, OpenGLTextureID );
        }
    }
    // OpenCLBuffer
    // Encapsulates an OpenCL ComputeBuffer and a host-side buffer. The data may exist on
    // the host and/or the device. Copying data between host and device is implemented in
    // methods CopyToDevice and CopyFromDevice. Access of the host-side data is supported
    // using the [] operator.
    public class OpenCLBuffer<T> where T : struct
    {
        // CPU/GPU buffer - wrapper around ComputerBuffer<T> and T[], based on Duncan Ogilvie
        ComputeCommandQueue _queue;
        T[] _cpubuffer;
        public ComputeBuffer<T> _gpubuffer;
        public const int ON_DEVICE = 1;
        public const int ON_HOST = 2;
        public const int WRITE_ONLY = 4;
        public const int READ_ONLY = 8;
        public const int READ_WRITE = 16;
        public OpenCLBuffer( OpenCLProgram ocl, int length, int flags = ON_DEVICE + ON_HOST + READ_WRITE )
        {
            _queue = ocl.queue;
            int clflags = 0;
            if ((flags & READ_ONLY) > 0) clflags += (int)ComputeMemoryFlags.ReadOnly;
            if ((flags & WRITE_ONLY) > 0) clflags += (int)ComputeMemoryFlags.WriteOnly;
            if ((flags & READ_WRITE) > 0) clflags += (int)ComputeMemoryFlags.ReadWrite;
            if ((flags & ON_HOST) > 0)
            {
                _cpubuffer = new T[length];
                clflags += (int)ComputeMemoryFlags.UseHostPointer;
            }
            if ((flags & ON_DEVICE) > 0)
            {
                _gpubuffer = new ComputeBuffer<T>( ocl.context, (ComputeMemoryFlags)clflags, _cpubuffer );
                if ((flags & ON_HOST) > 0) CopyToDevice();
            }
        }
        public OpenCLBuffer( OpenCLProgram ocl, T[] buffer, int flags = ON_DEVICE + ON_HOST + READ_WRITE )
        {
            _queue = ocl.queue;
            int clflags = (int)ComputeMemoryFlags.UseHostPointer;
            if ((flags & READ_ONLY) > 0) clflags += (int)ComputeMemoryFlags.ReadOnly;
            if ((flags & WRITE_ONLY) > 0) clflags += (int)ComputeMemoryFlags.WriteOnly;
            if ((flags & READ_WRITE) > 0) clflags += (int)ComputeMemoryFlags.ReadWrite;
            _cpubuffer = buffer;
            if ((flags & ON_DEVICE) > 0)
            {
                _gpubuffer = new ComputeBuffer<T>( ocl.context, (ComputeMemoryFlags)clflags, _cpubuffer );
                CopyToDevice();
            }
        }
        public void CopyToDevice()
        {
            _queue.WriteToBuffer( _cpubuffer, _gpubuffer, true, null );
        }
        public void CopyFromDevice()
        {
            _queue.ReadFromBuffer( _gpubuffer, ref _cpubuffer, true, null );
        }
        public T this[int index]
        {
            get { return _cpubuffer[index]; }
            set { _cpubuffer[index] = value; }
        }
        public int Length { get { return _cpubuffer.Length; } }
    }
    // OpenCLProgram
    // Encapsulates the OpenCL context and queue and program. The constructor prepares
    // these, compiles the specified source file and attempts to initialize the GL interop
    // functionality.
    public class OpenCLProgram
    {
        public ComputeContext context;
        public ComputeCommandQueue queue;
        public ComputeProgram program;
        public bool GLInteropAvailable = false;
        [System.Runtime.InteropServices.DllImport("opengl32", SetLastError = true)] static extern IntPtr wglGetCurrentDC();
        public void FatalError( string message )
        {
            MessageBox.Show( message, "OpenCL Error", MessageBoxButtons.OK, MessageBoxIcon.Error );
            System.Environment.Exit(1);
        }
        public void SelectBestDevice()
        {
            // This function attempts to find the best platform / device for OpenCL code execution.
            // The best device is typically not the CPU, nor an integrated GPU. If no GPU is found,
            // the CPU will be used, but this may limit compatibility, especially for the interop
            // functionality, but sometimes also for floating point textures.
            int bestPlatform = -1, bestDevice = -1, bestScore = -1;
            for( int i = 0; i < ComputePlatform.Platforms.Count; i++ )
            {
                var platform = ComputePlatform.Platforms[i];
                for( int j = 0; j < platform.Devices.Count; j++ )
                {
                    var device = platform.Devices[j];
                    if (device.Type == ComputeDeviceTypes.Gpu)
                    {
                        // found a gpu device; prefer this over integrated graphics
                        int score = 1;
                        if (!platform.Name.Contains( "Intel" )) score = 10; // AMD or NVidia
                        if (score > bestScore)
                        {
                            bestPlatform = i;
                            bestDevice = j;
                            bestScore = score;
                        }
                    }
                    else if (bestPlatform == -1)
                    {
                        // found an OpenCL device, but not a gpu, better than nothing
                        bestPlatform = i;
                        bestDevice = j;
                    }
                }
            }
            if (bestPlatform > -1)
            {
                var platform = ComputePlatform.Platforms[bestPlatform];
                Console.Write( "initializing OpenCL... " + platform.Name + " (" + platform.Profile + ").\n" );
                // try to enable gl interop functionality
                try
                {
                    var ctx = (OpenTK.Graphics.IGraphicsContextInternal)OpenTK.Graphics.GraphicsContext.CurrentContext;
                    IntPtr glHandle = ctx.Context.Handle;
                    IntPtr wglHandle = wglGetCurrentDC();
                    var p1 = new ComputeContextProperty( ComputeContextPropertyName.Platform, platform.Handle.Value );
                    var p2 = new ComputeContextProperty( ComputeContextPropertyName.CL_GL_CONTEXT_KHR, glHandle );
                    var p3 = new ComputeContextProperty( ComputeContextPropertyName.CL_WGL_HDC_KHR, wglHandle );
                    List<ComputeContextProperty> props = new List<ComputeContextProperty>() { p1, p2, p3 };
                    ComputeContextPropertyList Properties = new ComputeContextPropertyList( props );
                    context = new ComputeContext( ComputeDeviceTypes.Gpu, Properties, null, IntPtr.Zero );
                    GLInteropAvailable = true;
                }
                catch
                {
                    // if this failed, we'll do without gl interop
                    try
                    {
                        context = new ComputeContext( ComputeDeviceTypes.Gpu, new ComputeContextPropertyList( platform ), null, IntPtr.Zero );
                    }
                    catch
                    {
                        // failed to initialize a valid context; report
                        FatalError( "Failed to initialize OpenCL context" );
                    }
                }
            }
            else
            {
                // failed to find an OpenCL device; report
                FatalError( "Failed to initialize OpenCL" );
            }
        }
        public OpenCLProgram( string sourceFile )
        {
            // pick first platform
            SelectBestDevice();
            // load opencl source
            string clSource = "";
            try
            {
                var streamReader = new StreamReader( sourceFile );
                clSource = streamReader.ReadToEnd();
                streamReader.Close();
            }
            catch
            {
                FatalError( "File not found:\n" + sourceFile );
            }
            // create program with opencl source
            program = new ComputeProgram( context, clSource );
            // compile opencl source
            try
            {
                program.Build( null, null, null, IntPtr.Zero );
            }
            catch
            {
                FatalError( "Error in kernel code:\n" + program.GetBuildLog( context.Devices[0] ) );
            }
            // create a command queue with first gpu found
            queue = new ComputeCommandQueue( context, context.Devices[0], 0 );
        }
    }
    // OpenCL kernel
    // Encapsulates an OpenCL kernel. Multiple kernels may exist in the same program.
    // SetArgument methods are provided to conveniently set various argument types.
    public class OpenCLKernel
    {
        ComputeKernel kernel;
        ComputeCommandQueue queue;
        public OpenCLKernel( OpenCLProgram ocl, string kernelName )
        {
            // make a copy of the queue descriptor
            queue = ocl.queue;
            // load chosen kernel from program
            kernel = ocl.program.CreateKernel( kernelName );
        }
        public void SetArgument<T>( int i, T v ) where T : struct { kernel.SetValueArgument( i, v ); }
        public void SetArgument( int i, ComputeBuffer<int> v ) { kernel.SetMemoryArgument( i, v ); }
        public void SetArgument( int i, ComputeBuffer<float> v ) { kernel.SetMemoryArgument( i, v ); }
        public void SetArgument( int i, ComputeBuffer<uint> v ) { kernel.SetMemoryArgument( i, v ); }
        public void SetArgument( int i, OpenCLBuffer<int> v ) { kernel.SetMemoryArgument( i, v._gpubuffer ); }
        public void SetArgument( int i, OpenCLBuffer<float> v ) { kernel.SetMemoryArgument( i, v._gpubuffer ); }
        public void SetArgument( int i, OpenCLBuffer<uint> v ) { kernel.SetMemoryArgument( i, v._gpubuffer ); }
        public void SetArgument( int i, ComputeImage2D v ) { kernel.SetMemoryArgument( i, v ); }
        public void SetArgument( int i, OpenCLImage<int> v ) { kernel.SetMemoryArgument( i, v.texBuffer ); }
        public void SetArgument( int i, OpenCLImage<float> v ) { kernel.SetMemoryArgument( i, v.texBuffer ); }
        public void LockOpenGLObject( ComputeImage2D image )
        {
            List<ComputeMemory> c = new List<ComputeMemory>() { image };
            queue.AcquireGLObjects( c, null );
        }
        public void UnlockOpenGLObject( ComputeImage2D image )
        {
            queue.Finish();
            List<ComputeMemory> c = new List<ComputeMemory>() { image };
            queue.ReleaseGLObjects( c, null );
        }
        public void Execute( long [] workSize )
        {
            queue.Execute( kernel, null, workSize, null, null );
        }
        public void Execute( long [] workSize, long [] localSize )
        {
            queue.Execute( kernel, null, workSize, localSize, null );
        }
    }
}

