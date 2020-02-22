using Cloo;
using OpenTK.Input;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.IO;


namespace Template {
    class Life {
        int generatie, aantal_cellen;
        int2 size, size_in_ints;

#region opencl variabelen
        bool GLInterop = true;

        static OpenCLProgram ocl = new OpenCLProgram("../../src/opencl/program.cl");
        OpenCLKernel k_sim  = new OpenCLKernel(ocl, "next");
        OpenCLKernel k_copy = new OpenCLKernel(ocl, "copy_cells");
        
        OpenCLBuffer<uint> patroon, volgende;
        OpenCLBuffer<int> buffer;
        OpenCLImage <int> image;

        long[] werk, werk_klein;
        long[] lokaalWerk = new long[] {16, 16};
#endregion

#region Invoer
        int scrollValue, lastScrollValue, lastMouseWheel, maxScroll = 7;
        float zoom, lastZoom;
        int2 resolution, resolutionStart, lastResolution;

        int2 offset = new int2(0,0);
        int2 dragStart, offsetStart;
        bool lastLButtonState = false;
#endregion

        public Surface screen;
        Stopwatch timer = new Stopwatch();

        public void Init() {
            timer.Start(); Console.WriteLine("Initialiseren...");

            ReadGoLFile(); 
            patroon.CopyToDevice(); //Kopieer het patroon naar de gpu, waar het zal blijven
            
            timer.Stop(); Console.WriteLine("Patroon geladen: ({0}x{1}) in {2}s", size.x, size.y, timer.ElapsedMilliseconds*0.001f);
            timer.Reset();

            resolutionStart = new int2(screen.width, screen.height);
            resolution      = resolutionStart;
            lastResolution  = resolution;

            k_sim.SetArgument(0, volgende);
            k_sim.SetArgument(1, patroon);
            k_sim.SetArgument(2, size_in_ints);
            k_sim.SetArgument(3, resolution);
            k_sim.SetArgument(4, offset);

            k_copy.SetArgument(0, volgende);
            k_copy.SetArgument(1, patroon);
            k_copy.SetArgument(2, size_in_ints.x);
            k_copy.SetArgument(3, size_in_ints.y);
        }

        public void Process() {
            GL.Finish();
            screen.Clear(0);
            timer.Start();

            generatie++;
            // if (generatie % 2 == 1) {
            //     k_sim.SetArgument(0, patroon);
            //     k_sim.SetArgument(1, volgende); // @Todo: dit ook voor de halo kernels
            //     k_copy.SetArgument(0, patroon);
            //     k_copy.SetArgument(1, volgende); 
            // }
            // else {
            //     k_sim.SetArgument(0, volgende);
            //     k_sim.SetArgument(1, patroon);
            //     k_copy.SetArgument(0, volgende);
            //     k_copy.SetArgument(1, patroon); 
            // }

            if (GLInterop) {
                if (resolution != lastResolution) {
                    image = new OpenCLImage<int>(ocl, resolution.x, resolution.y);
                    lastResolution = resolution;
                    k_sim.SetArgument(3, resolution);
                }

                k_sim.SetArgument(5, image);
                k_sim.LockOpenGLObject(image.texBuffer);
                k_sim.Execute(werk);
                k_sim.UnlockOpenGLObject(image.texBuffer);
            }
            else {
                k_sim.SetArgument(5, buffer);
                k_sim.Execute(werk);
                buffer.CopyFromDevice();
                for (int i = 0; i < buffer.Length; ++i) screen.pixels[i] = buffer[i];
            }

            k_copy.Execute(werk_klein);  // Kopieer wat zojuist in patroon is geschreven naar de tweede buffer

            timer.Stop();
            Console.Write("\r{0}ms", timer.ElapsedMilliseconds);
            timer.Reset();
        }

#region Invoer
        public void SetMouseState(OpenTK.Input.MouseState m, int x, int y) {
            if (m.LeftButton == ButtonState.Pressed) {
                if (lastLButtonState) {
                    int2   d = zoom * new int2((x - dragStart.x), y - dragStart.y);
                    offset.x = Math.Min(size.x - (int)(screen.width *zoom), Math.Max( 0, offsetStart.x - d.x));
                    offset.y = Math.Min(size.y - (int)(screen.height*zoom), Math.Max( 0, offsetStart.y - d.y));
                    k_sim.SetArgument(4, offset);
                }
                else {
                    dragStart   = new int2(x, y);
                    offsetStart = new int2(offset.x, offset.y);
                    lastLButtonState = true;
                }
            }
            else lastLButtonState = false;
            
            // Zoom
            scrollValue += -(m.Wheel - lastMouseWheel);
            lastMouseWheel = m.Wheel;
            scrollValue = Math.Min(maxScroll, Math.Max(scrollValue, -maxScroll));
            zoom = 1 + (scrollValue+1)/(float)maxScroll;
 
            if (scrollValue != lastScrollValue) {
                lastScrollValue = scrollValue;
                resolution = zoom * resolutionStart;
            }
        }
#endregion

        public void ReadGoLFile() {
            StreamReader sr = new StreamReader("../../maps/turing_js_r.rle");
            uint state = 0, n = 0, x = 1, y = 1;
            
            while (true) {
                String line = sr.ReadLine();
                if (line == null) break; // end of file
                int pos = 0;
                if (line[pos] == '#') continue; /* comment line */
                else if (line[pos] == 'x') { //Header
                    String[] sub = line.Split(new char[] { '=', ',' }, StringSplitOptions.RemoveEmptyEntries);

                    size = new int2(Int32.Parse(sub[1]), Int32.Parse(sub[3]));
                    size_in_ints = new int2((size.x + 31) / 32 + 2, size.y + 2); // Voeg padding toe
                    int lengte_in_ints = size_in_ints.x * size_in_ints.y;

                    // Console.WriteLine("Grootte van arrays: {0}x{1}", size_in_ints.x*32, size_in_ints.y);

                    patroon  = new OpenCLBuffer<uint>(ocl, lengte_in_ints);
                    volgende = new OpenCLBuffer<uint>(ocl, lengte_in_ints);
                    image   = new OpenCLImage <int> (ocl, screen.width, screen.height);
                    buffer  = new OpenCLBuffer<int> (ocl, screen.width * screen.height);
                    
                    werk       = new long[] { size.x, size.y };
                    werk_klein = new long[] { lengte_in_ints };
                }
                else while (pos < line.Length) {
                    Char c = line[pos++];
                    if (state == 0) if (c < '0' || c > '9') { state = 1; n = Math.Max(n, 1); } else n = (uint)(n * 10 + (c - '0'));
                    if (state == 1) { // expect other character
                        if (c == '$') { y += n; x = 1; } // newline
                        else if (c == 'o') 
                            for (int i = 0; i < n; i++) {
                                patroon[(int)((y ) * size_in_ints.x + ((x >> 5) ))] |= 1U << (int)(x & 31);
                                x++;
                            }
                        else if (c == 'b') x += n;
                        state = n = 0;
                    }
                }
            }
        }

        public void Render() {
            if (GLInterop) {
                GL.LoadIdentity();
                GL.BindTexture( TextureTarget.Texture2D, image.OpenGLTextureID );
                GL.Begin( PrimitiveType.Quads );
                GL.TexCoord2( 0.0f, 1.0f ); GL.Vertex2( -1.0f, -1.0f );
                GL.TexCoord2( 1.0f, 1.0f ); GL.Vertex2(  1.0f, -1.0f );
                GL.TexCoord2( 1.0f, 0.0f ); GL.Vertex2(  1.0f,  1.0f );
                GL.TexCoord2( 0.0f, 0.0f ); GL.Vertex2( -1.0f,  1.0f );
                GL.End();
            }
        }
    }
}

