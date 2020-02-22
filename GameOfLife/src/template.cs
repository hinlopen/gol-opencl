using System;
using System.Drawing;
using System.Runtime.InteropServices;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace Template {
    internal static class CursorPosition {
        [StructLayout(LayoutKind.Sequential)]
        public struct POINT {
            public int X;
            public int Y;
            public static implicit operator Point(POINT point) { return new Point(point.X, point.Y); }
        }
        [DllImport("user32.dll")]
        public static extern bool GetCursorPos(out POINT lpPoint);
        public static Point GetCursorPosition() {
            POINT lpPoint;
            GetCursorPos(out lpPoint);
            return lpPoint;
        }
    }

    // OpenTKApp
    // Overloads the OpenTK GameWindow class. The template creates a single OpenGL texture,
    // identified by screenID. The pixels of this texture come from Surface game.screen and
    // are uploaded to the GPU after every game.Tick(), providing a linear pixel buffer to
    // the game class.
    // After rendering a full-screen quad using this texture, the template executes
    // game.Render(), which may perform additional OpenGL rendering on top of the initial
    // output.
    public class OpenTKApp : GameWindow
    {
        static int screenID;
        static Life game;
        protected override void OnLoad( EventArgs e )
        {
            // called upon app init
            GL.ClearColor( Color.Black );
            GL.Enable( EnableCap.Texture2D );
            GL.Hint( HintTarget.PerspectiveCorrectionHint, HintMode.Nicest );
            Width  = 600;
            Height = 600;
            X = DisplayDevice.Default.Width  / 2 - Width / 2;
            Y = DisplayDevice.Default.Height / 2 - Height / 2;
            game = new Life();
            game.screen = new Surface( 600, 600 );
            Sprite.target = game.screen;
            screenID = game.screen.GenTexture();
            game.Init();
        }
        protected override void OnUnload( EventArgs e )
        {
            // called upon app close
            GL.DeleteTextures( 1, ref screenID );
            Environment.Exit( 0 ); // bypass wait for key on CTRL-F5
        }
        protected override void OnResize( EventArgs e )
        {
            // called upon window resize
            GL.Viewport(0, 0, Width, Height);
            GL.MatrixMode( MatrixMode.Projection );
            GL.LoadIdentity();
            GL.Ortho( -1.0, 1.0, -1.0, 1.0, 0.0, 4.0 );
        }
        protected override void OnUpdateFrame( FrameEventArgs e )
        {
            // called once per frame; app logic
            var keyboard = OpenTK.Input.Keyboard.GetState();
            if (keyboard[OpenTK.Input.Key.Escape]) this.Exit();
            var mouse = OpenTK.Input.Mouse.GetState();
            Point p = CursorPosition.GetCursorPosition();
            if (p.X > X && p.X <= X + Width && p.Y > Y && p.Y <= p.Y + Width)
                game.SetMouseState(mouse, p.X, p.Y);
            
        }
        protected override void OnRenderFrame( FrameEventArgs e )
        {
            // called once per frame; render
            game.Process();
            GL.BindTexture( TextureTarget.Texture2D, screenID );
            GL.TexImage2D( TextureTarget.Texture2D,
                           0,
                           PixelInternalFormat.Rgba,
                           game.screen.width,
                           game.screen.height,
                           0,
                           OpenTK.Graphics.OpenGL.PixelFormat.Bgra,
                           PixelType.UnsignedByte,
                           game.screen.pixels
                         );
            GL.Clear( ClearBufferMask.ColorBufferBit );
            GL.MatrixMode( MatrixMode.Modelview );
            GL.LoadIdentity();
            GL.BindTexture( TextureTarget.Texture2D, screenID );
            GL.Begin( PrimitiveType.Quads );
            GL.TexCoord2( 0.0f, 1.0f ); GL.Vertex2( -1.0f, -1.0f );
            GL.TexCoord2( 1.0f, 1.0f ); GL.Vertex2(  1.0f, -1.0f );
            GL.TexCoord2( 1.0f, 0.0f ); GL.Vertex2(  1.0f,  1.0f );
            GL.TexCoord2( 0.0f, 0.0f ); GL.Vertex2( -1.0f,  1.0f );
            GL.End();
            game.Render();
            SwapBuffers();
        }

        [STAThread]
        public static void Main()
        {
            // entry point
            using (OpenTKApp app = new OpenTKApp())
            {
                app.Run( 30.0, 0.0 );
            }
        }
    }
    // float2
    // Basic vector class, designed to be mostly compatible with OpenCL. Unsafe, to
    // enable the use of a C++ 'union' via FieldOffsets and the [] operator. A vector
    // can now be accessed using x, y, but also using [0] and [1]. Be carefull:
    // out-of-bounds accesses are not checked (hence: unsafe).
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct float2
    {
        [FieldOffset(0)]
        public float x;
        [FieldOffset(4)]
        public float y;
        [FieldOffset(0)]
        public fixed float data[2];
        public float2( float a, float b ) { x = a; y = b; }
        public float2( float a ) { x = y = a; }
        public static float2 operator - ( float2 a ) { return new float2( -a.x, -a.y ); }
        public static float2 operator + ( float2 a, float2 b ) { return new float2( a.x + b.x, a.y + b.y ); }
        public static float2 operator * ( float2 a, float s ) { return new float2( s * a.x, s * a.y ); }
        public static float2 operator * ( float s, float2 a ) { return new float2( s * a.x, s * a.y ); }
        public static float2 operator / ( float2 a, float s ) { float r = 1.0f / s; return new float2( a.x * r, a.y * r ); }
        public static float2 operator - ( float2 a, float2 b ) { return new float2( a.x - b.x, a.y - b.y ); }
        public static float length( float2 a ) { return a.length(); }
        public float length() { return (float)Math.Sqrt( x * x + y * y ); }
        public float2 normalize() { float r = 1.0f / length(); x *= r; y *= r; return this; }
        public static float2 normalize( float2 a ) { return a.normalize(); }
        public static float dot( float2 a, float2 b ) { return a.x * b.x + a.y * b.y; }
        public float dot( float2 b ) { return x * b.x + y * b.y; }
        public float this[int idx]
        {
            get { unsafe { fixed( float* p = data) { return p[idx]; } } }
            set { unsafe { fixed( float* p = data) { p[idx] = value; } } }
        }
    }
    // float3
    // Basic vector class, designed to be mostly compatible with OpenCL.
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct float3
    {
        [FieldOffset(0)]
        public float x;
        [FieldOffset(4)]
        public float y;
        [FieldOffset(8)]
        public float z;
        [FieldOffset(12)]
        public float dummy;
        [FieldOffset(0)]
        public fixed float data[3];
        public float3( float a, float b, float c ) { x = a; y = b; z = c; dummy = 0; }
        public float3( float a ) { x = y = z = a; dummy = 0; }
        public static float3 operator - ( float3 a ) { return new float3( -a.x, -a.y, -a.z ); }
        public static float3 operator + ( float3 a, float3 b ) { return new float3( a.x + b.x, a.y + b.y, a.z + b.z ); }
        public static float3 operator * ( float3 a, float s ) { return new float3( s * a.x, s * a.y, s * a.z ); }
        public static float3 operator * ( float s, float3 a ) { return new float3( s * a.x, s * a.y, s * a.z ); }
        public static float3 operator / ( float3 a, float s ) { float r = 1.0f / s; return new float3( a.x * r, a.y * r, a.z * r ); }
        public static float3 operator - ( float3 a, float3 b ) { return new float3( a.x - b.x, a.y - b.y, a.z - b.z ); }
        public static float length( float3 a ) { return a.length(); }
        public float length() { return (float)Math.Sqrt( x * x + y * y + z * z ); }
        public float3 normalize() { float r = 1.0f / length(); x *= r; y *= r; z *= r; return this; }
        public static float3 normalize( float3 a ) { return a.normalize(); }
        public static float dot( float3 a, float3 b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        public float dot( float3 b ) { return x * b.x + y * b.y + z * b.z; }
        public float3 cross( float3 b ) { return new float3( y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x ); }
        public static float3 cross( float3 a, float3 b ) { return a.cross( b ); }
        public float this[int idx]
        {
            get { unsafe { fixed( float* p = data) { return p[idx]; } } }
            set { unsafe { fixed( float* p = data) { p[idx] = value; } } }
        }
    }
    // float4
    // Basic vector class, designed to be mostly compatible with OpenCL.
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct float4
    {
        [FieldOffset(0)]
        public float x;
        [FieldOffset(4)]
        public float y;
        [FieldOffset(8)]
        public float z;
        [FieldOffset(12)]
        public float w; // note: 3-component types in OpenCL take 4 * sizeof(component)
        [FieldOffset(0)]
        public fixed float data[4];
        public float4( float a, float b, float c, float d ) { x = a; y = b; z = c; w = d; }
        public float4( float a ) { x = y = z = w = a; }
        public float4( float3 a, float b ) { x = a.x; y = a.y; z = a.z; w = b; }
        public float4( float2 a, float2 b ) { x = a.x; y = a.y; z = b.x; w = b.y; }
        public static float4 operator - ( float4 a ) { return new float4( -a.x, -a.y, -a.z, -a.w ); }
        public static float4 operator + ( float4 a, float4 b ) { return new float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }
        public static float4 operator * ( float4 a, float s ) { return new float4( s * a.x, s * a.y, s * a.z, s * a.w ); }
        public static float4 operator * ( float s, float4 a ) { return new float4( s * a.x, s * a.y, s * a.z, s * a.w ); }
        public static float4 operator / ( float4 a, float s ) { float r = 1.0f / s; return new float4( a.x * r, a.y * r, a.z * r, a.w * r ); }
        public static float4 operator - ( float4 a, float4 b ) { return new float4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); }
        public static float length( float4 a ) { return a.length(); }
        public float length() { return (float)Math.Sqrt( x * x + y * y + z * z + w * w ); }
        public float4 normalize() { float r = 1.0f / length(); x *= r; y *= r; z *= r; w *= r; return this; }
        public static float4 normalize( float4 a ) { return a.normalize(); }
        public static float dot( float4 a, float4 b ) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
        public float dot( float4 b ) { return x * b.x + y * b.y + z * b.z + w * b.w; }
        public float this[int idx]
        {
            get { unsafe { fixed( float* p = data) { return p[idx]; } } }
            set { unsafe { fixed( float* p = data) { p[idx] = value; } } }
        }
    }
    // int2
    // Basic vector class, designed to be mostly compatible with OpenCL. Unsafe, to
    // enable the use of a C++ 'union' via FieldOffsets and the [] operator. A vector
    // can now be accessed using x, y, but also using [0] and [1]. Be carefull:
    // out-of-bounds accesses are not checked (hence: unsafe).
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct int2
    {
        [FieldOffset(0)]
        public int x;
        [FieldOffset(4)]
        public int y;
        [FieldOffset(0)]
        public fixed int data[2];
        public int2( int a, int b ) { x = a; y = b; }
        public int2( int a ) { x = y = a; }
        public static int2 operator - ( int2 a ) { return new int2( -a.x, -a.y ); }
        public static int2 operator + ( int2 a, int2 b ) { return new int2( a.x + b.x, a.y + b.y ); }
        public static int2 operator * ( int2 a, int s ) { return new int2( s * a.x, s * a.y ); }
        public static int2 operator * ( float a, int2 s ) { return new int2( (int)(a * s.x), (int)(a * s.y)); }
        public static int2 operator * ( int s, int2 a ) { return new int2( s * a.x, s * a.y ); }
        public static int2 operator / ( int2 a, int s ) { return new int2( a.x / s, a.y / s ); }
        public static int2 operator - ( int2 a, int2 b ) { return new int2( a.x - b.x, a.y - b.y ); }
        public static bool operator == ( int2 a, int2 b ) { return a.x == b.x && a.y == b.y; }
        public static bool operator != ( int2 a, int2 b ) { return !(a == b);  }
        public static int dot( int2 a, int2 b ) { return a.x * b.x + a.y * b.y; }
        public int dot( int2 b ) { return x * b.x + y * b.y; }
        public int this[int idx]
        {
            get { unsafe { fixed( int* p = data) { return p[idx]; } } }
            set { unsafe { fixed( int* p = data) { p[idx] = value; } } }
        }
    }
    // int3
    // Basic vector class, designed to be mostly compatible with OpenCL. Unsafe, to
    // enable the use of a C++ 'union' via FieldOffsets and the [] operator. A vector
    // can now be accessed using x, y, but also using [0] and [1]. Be carefull:
    // out-of-bounds accesses are not checked (hence: unsafe).
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct int3
    {
        [FieldOffset(0)]
        public int x;
        [FieldOffset(4)]
        public int y;
        [FieldOffset(8)]
        public int z;
        [FieldOffset(12)]
        public int dummy; // note: 3-component types in OpenCL take 4 * sizeof(component)
        [FieldOffset(0)]
        public fixed int data[3];
        public int3( int a, int b, int c ) { x = a; y = b; z = c; dummy = 0; }
        public int3( int a ) { x = y = a; z = a; dummy = 0; }
        public static int3 operator - ( int3 a ) { return new int3( -a.x, -a.y, -a.z ); }
        public static int3 operator + ( int3 a, int3 b ) { return new int3( a.x + b.x, a.y + b.y, a.z + b.z ); }
        public static int3 operator * ( int3 a, int s ) { return new int3( s * a.x, s * a.y, s * a.z ); }
        public static int3 operator * ( int s, int3 a ) { return new int3( s * a.x, s * a.y, s * a.z ); }
        public static int3 operator / ( int3 a, int s ) { return new int3( a.x / s, a.y / s, a.z / s ); }
        public static int3 operator - ( int3 a, int3 b ) { return new int3( a.x - b.x, a.y - b.y, a.z - b.z ); }
        public static int dot( int3 a, int3 b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        public int dot( int3 b ) { return x * b.x + y * b.y + z * b.z; }
        public int this[int idx]
        {
            get { unsafe { fixed( int* p = data) { return p[idx]; } } }
            set { unsafe { fixed( int* p = data) { p[idx] = value; } } }
        }
    }
    // int4
    // Basic vector class, designed to be mostly compatible with OpenCL. Unsafe, to
    // enable the use of a C++ 'union' via FieldOffsets and the [] operator. A vector
    // can now be accessed using x, y, but also using [0] and [1]. Be carefull:
    // out-of-bounds accesses are not checked (hence: unsafe).
    [StructLayout(LayoutKind.Explicit)]
    public unsafe struct int4
    {
        [FieldOffset(0)]
        public int x;
        [FieldOffset(4)]
        public int y;
        [FieldOffset(8)]
        public int z;
        [FieldOffset(12)]
        public int w;
        [FieldOffset(0)]
        public fixed int data[4];
        public int4( int a, int b, int c, int d ) { x = a; y = b; z = c; w = d; }
        public int4( int a ) { x = y = a; z = a; w = a; }
        public static int4 operator - ( int4 a ) { return new int4( -a.x, -a.y, -a.z, -a.w ); }
        public static int4 operator + ( int4 a, int4 b ) { return new int4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w ); }
        public static int4 operator * ( int4 a, int s ) { return new int4( s * a.x, s * a.y, s * a.z, s * a.w ); }
        public static int4 operator * ( int s, int4 a ) { return new int4( s * a.x, s * a.y, s * a.z, s * a.w ); }
        public static int4 operator / ( int4 a, int s ) { return new int4( a.x / s, a.y / s, a.z / s, a.w / s ); }
        public static int4 operator - ( int4 a, int4 b ) { return new int4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w ); }
        public static int dot( int4 a, int4 b ) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
        public int dot( int4 b ) { return x * b.x + y * b.y + z * b.z + w * b.w; }
        public int this[int idx]
        {
            get { unsafe { fixed( int* p = data) { return p[idx]; } } }
            set { unsafe { fixed( int* p = data) { p[idx] = value; } } }
        }
    }
}
