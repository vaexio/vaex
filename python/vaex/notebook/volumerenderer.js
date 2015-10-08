(function($) {
var colormap_names = ["PaulT_plusmin", "binary", "Blues", "BuGn", "BuPu", "gist_yarg", "GnBu", "Greens", "Greys", "Oranges", "OrRd", "PuBu", "PuBuGn", "PuRd", "Purples", "RdPu", "Reds", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd", "afmhot", "autumn", "bone", "cool", "copper", "gist_gray", "gist_heat", "gray", "hot", "pink", "spring", "summer", "winter", "BrBG", "bwr", "coolwarm", "PiYG", "PRGn", "PuOr", "RdBu", "RdGy", "RdYlBu", "RdYlGn", "seismic", "Accent", "Dark2", "hsv", "Paired", "Pastel1", "Pastel2", "Set1", "Set2", "Set3", "spectral", "gist_earth", "gist_ncar", "gist_rainbow", "gist_stern", "jet", "brg", "CMRmap", "cubehelix", "gnuplot", "gnuplot2", "ocean", "rainbow", "terrain", "flag", "prism"];

    var gl;
    function initGL(canvas) {
			console.log(window.WebGLRenderingContext)
			console.log(canvas.getContext("webkit-3d"))
            gl = canvas.getContext("experimental-webgl");
            if(gl) {
				gl.viewportWidth = canvas.width;
				gl.viewportHeight = canvas.height;
			}
        if (!gl) {
            alert("Could not initialise WebGL, sorry :-(");
        }
    }
    function loadShader(url, name) {
        assign = function(response) {
            //console.log("name " + name +" = " + response)
            shader_text[name] = response
            console.log("shader: " +name)
            shaders_loaded += 1;
        }
        return $.ajax({
            url: url,
        }).then(assign)
    }

    var shader_text = {}
    var shader_names = ["cube", "texture", "volr"]
    var shaders_loaded = 0;
    function are_shaders_loaded() {
        return true; //shader_text.length == shader_names.length * 2
    }
    function loadShaders(base_url) {
        for(var i = 0; i < shader_names.length; i++) {
            loadShader(base_url + shader_names[i] + "-fragment.shader", shader_names[i]+"-fragment")
            loadShader(base_url + shader_names[i] + "-vertex.shader", shader_names[i]+"-vertex")
        }

    }

    //loadShaders("/nbextensions/volr/")


    function getShader_(gl, id, replacements) {
        var shaderScript = document.getElementById(id);
        if (!shaderScript) {
            alert("Cannot find element " + id);
            return null;
        }

        var str = "";
        var k = shaderScript.firstChild;
        while (k) {
            if (k.nodeType == 3) {
                str += k.textContent;
            }
            k = k.nextSibling;
        }
        if(replacements) {
			console.log(replacements)
			for(var key in replacements) {
				//console.log(replacements)
				str  = str.replace(key, replacements[key])
			}
		}

        var shader;
        if (shaderScript.type == "x-shader/x-fragment") {
            shader = gl.createShader(gl.FRAGMENT_SHADER);
        } else if (shaderScript.type == "x-shader/x-vertex") {
            shader = gl.createShader(gl.VERTEX_SHADER);
        } else {
            return null;
        }

        gl.shaderSource(shader, str);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(shader));
            return null;
        }

        return shader;
    }

    function getShader(gl, id, replacements) {
        console.log("get shader " + id)
        //var str = shader_text[id]
        var str = window.shader_cache[id]
        console.log("id = " + id)
        console.log(str)
        if(replacements) {
			console.log(replacements)
			for(var key in replacements) {
				//console.log(replacements)
				str  = str.replace(key, replacements[key])
			}
		}

        var shader;
        //if (shaderScript.type == "x-shader/x-fragment") {
        if(id.indexOf("fragment") != -1) {
            shader = gl.createShader(gl.FRAGMENT_SHADER);
            console.log("fragment shader")
        } else if(id.indexOf("vertex") != -1) {
            shader = gl.createShader(gl.VERTEX_SHADER);
            console.log("vertex shader")
        } else {
            alert("no shader " +id)
            return null;
        }

        gl.shaderSource(shader, str);
        gl.compileShader(shader);

        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(shader));
            return null;
        }

        return shader;
    }

    function initShaders(name, replacements) {
		var fragmentShader = getShader(gl, name+"_fragment", replacements);
		var vertexShader = getShader(gl, name+"_vertex", replacements);
		var program = gl.createProgram();
		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);

		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			alert("Could not initialise shaders");
		}

		program.pMatrixUniform = gl.getUniformLocation(program, "uPMatrix");
		program.mvMatrixUniform = gl.getUniformLocation(program, "uMVMatrix");
        program.vertexPositionAttribute = gl.getAttribLocation(program, "aVertexPosition");
        gl.enableVertexAttribArray(program.vertexPositionAttribute);

		return program;
	}

    $.vr = function(canvas, options) {
        var plugin = this;

        initGL(canvas);

        function initVolumeTexture(src) {
            var texture_volume = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, texture_volume);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, volume_size, volume_size, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            loadTexture(texture_volume, gl.RGBA, src);
            return texture_volume

            //var ext = gl.getExtension("OES_texture_float")
            //console.log("ext:" + ext.FLOAT);
        }

        var volume_size = 128;
        function initColormapTexture(src) {
            var texture_colormaps = gl.createTexture();
            gl.bindTexture(gl.TEXTURE_2D, texture_colormaps);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, volume_size, volume_size, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            colormap_image = loadTexture(texture_colormaps, gl.RGB, src);
            return texture_colormaps;
            //var ext = gl.getExtension("OES_texture_float")
            //console.log("ext:" + ext.FLOAT);
        }

        function loadTexture(texture, format, url) {
            var textureImage = new Image();
            textureImage.onload = function() {
                gl.bindTexture(gl.TEXTURE_2D, texture);
                gl.texImage2D(gl.TEXTURE_2D, 0, format, format, gl.UNSIGNED_BYTE, textureImage);
                //alert("loaded: " +volumeImage.src + " " +gl.getError() + ":" +volumeImage);
                plugin.updateScene();
            }
            textureImage.src = url;
            return textureImage;

        }

        var transfer_function_array = []

        this.fill_transfer_function_array = function() {
          for(var i = 0; i < 1024; i++) {
            var position = i / (1023.);
            var intensity = 0.;
            for(var j = 0; j < 3; j++) {
              //float bla = length(sample)/sqrt(3.);
              //float data_value = () * data_scale;
              var chi = (position-options.volume_level[j])/options.volume_width[j];
              var chisq = Math.pow(chi, 2.);
              intensity += Math.exp(-chisq) * options.opacity[j];
            }
            //console.log(intensity)
            transfer_function_array.push([intensity * 255.])
          }
        }

        this.update_transfer_function_array = function(array) {
          gl.bindTexture(gl.TEXTURE_2D, texture_transfer_function);
          if(array == undefined) {
            plugin.fill_transfer_function_array()
            array = transfer_function_array
          } else {
            var proper_array = [];
            for(var i = 0; i < array.length; i++) {
              proper_array.push([array[i]*255])
            }
          }
          var transfer_function_uint8_array = new Uint8Array(array);
          console.log("array > " + array.length)
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.ALPHA, array.length, 1, 0, gl.ALPHA, gl.UNSIGNED_BYTE, transfer_function_uint8_array);

        }
        var shader_cube;
        var shader_texture;
        // 3 versions of the shaders exists
        var shader_volume_rendering_best;
        var shader_volume_rendering_fast;
        var shader_volume_rendering_poor;
        // used for different pruposes
        var shader_volume_rendering;
        var shader_volume_rendering_updates;
        var shader_volume_rendering_final;

        //loadShaders()
        if(!are_shaders_loaded()) {
            console.log("shaders not loaded, only" + shaders_loaded + " out of " +(shader_names.length*2))
        }
        var shader_cube = initShaders("cube");
        gl.useProgram(shader_cube);
        shader_texture = initShaders("texture");
        shader_volume_rendering_best = initShaders("volr", {NR_OF_STEPS:300});
        shader_volume_rendering_fast = initShaders("volr", {NR_OF_STEPS:80});
        shader_volume_rendering_poor = initShaders("volr", {NR_OF_STEPS:40});
        shader_volume_rendering = shader_volume_rendering_fast;
        shader_volume_rendering_updates = shader_volume_rendering_poor;
        shader_volume_rendering_final = shader_volume_rendering_best;

        var texture_volume = initVolumeTexture(options.cube)
        var texture_colormaps = initColormapTexture(options.colormap)

        var triangleVertexPositionBuffer;
        var squareVertexPositionBuffer;
        var cubeVertexPositionBuffer;
        var cubeIndexBuffer;
        var cubeColorBuffer;

        var frame_buffer;
        var texture_frame_buffer_front;
        var texture_frame_buffer_back;
        var texture_frame_buffer_volume;
        var texture_show;
        var texture_transfer_function;
        //var texture_colormaps;

        var texture_volume;
        var colormap_image;

        texture_transfer_function = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture_transfer_function);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        this.fill_transfer_function_array()
        var transfer_function_uint8_array = new Uint8Array(transfer_function_array);
        console.log("array > " + transfer_function_array.length)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.ALPHA, transfer_function_array.length, 1, 0, gl.ALPHA, gl.UNSIGNED_BYTE, transfer_function_uint8_array);
        console.log(gl.getError())

    		var frame_buffer = gl.createFramebuffer();
    		gl.bindFramebuffer(gl.FRAMEBUFFER, frame_buffer);
    		frame_buffer.width = options.frame_buffer_width;
    		frame_buffer.height = options.frame_buffer_height;


            // this is where we render the volume to
    		var texture_frame_buffer_volume = gl.createTexture();
    		gl.bindTexture(gl.TEXTURE_2D, texture_frame_buffer_volume);
    		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, frame_buffer.width, frame_buffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);


            // this is for the (x,y,z) coordinates encoded in rgb for the back plane
    		var texture_frame_buffer_back = gl.createTexture();
    		gl.bindTexture(gl.TEXTURE_2D, texture_frame_buffer_back);
    		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    		//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
    		//gl.generateMipmap(gl.TEXTURE_2D);
    		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, frame_buffer.width, frame_buffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

            // similar for the front
    		var texture_frame_buffer_front = gl.createTexture();
    		gl.bindTexture(gl.TEXTURE_2D, texture_frame_buffer_front);
    		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    		//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
    		//gl.generateMipmap(gl.TEXTURE_2D);
    		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, frame_buffer.width, frame_buffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

            // this is what we show, for debugging you may want to see the front or back
    		texture_show = texture_frame_buffer_volume;//texture_frame_buffer_back;
    		//texture_show = texture_frame_buffer_back;


    		var render_buffer = gl.createRenderbuffer();
    		gl.bindRenderbuffer(gl.RENDERBUFFER, render_buffer);
    		gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, frame_buffer.width, frame_buffer.height);

    		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture_frame_buffer_front, 0);
    		gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, render_buffer);

    		gl.bindTexture(gl.TEXTURE_2D, null);
    		gl.bindRenderbuffer(gl.RENDERBUFFER, null);
    		gl.bindFramebuffer(gl.FRAMEBUFFER, null);


        triangleVertexPositionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);
        var vertices = [
             0.0,  1.0,  0.0,
            -1.0, -1.0,  0.0,
             1.0, -1.0,  0.0
        ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        triangleVertexPositionBuffer.itemSize = 3;
        triangleVertexPositionBuffer.numItems = 3;

        // vertex buffers etc for cube and square
        squareVertexPositionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, squareVertexPositionBuffer);
        vertices = [
             1.0,  1.0,  0.0,
            -1.0,  1.0,  0.0,
             1.0, -1.0,  0.0,
            -1.0, -1.0,  0.0
        ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        squareVertexPositionBuffer.itemSize = 3;
        squareVertexPositionBuffer.numItems = 4;

        cubeVertexPositionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer);
        vertices = [
             0.0,  0.0,  0.0,
             1.0,  0.0,  0.0,
             1.0,  1.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0,
             1.0,  0.0,  1.0,
             1.0,  1.0,  1.0,
             0.0,  1.0,  1.0,
        ];
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        cubeVertexPositionBuffer.itemSize = 3;
        cubeVertexPositionBuffer.numItems = 8;

        // bind the positions to color as well
        cubeColorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeColorBuffer);
		    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        cubeColorBuffer.itemSize = 3;
        cubeColorBuffer.numItems = 8;

        cubeIndexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeIndexBuffer);
        /*

        3 2
        0 1

        7 6
        4 5
        */
        var indices = [
			0, 3, 2, // back
			0, 2, 1,
			5, 1, 2, // right
			5, 2, 6,
			4, 7, 3, // left
			4, 3, 0,
			7, 6, 2, // top
			7, 2, 3,
			5, 4, 0, // bottom
			5, 0, 1	,

			4, 5, 6, // front
			4, 6, 7
        ];
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);
        cubeIndexBuffer.itemSize = 1;
        cubeIndexBuffer.numItems = 3*2*6;//*4;



        var mvMatrix = mat4.create();
        var pMatrix = mat4.create();

        plugin.init = function() {
            log("ctor")
            log(this)
            log(canvas)

            //initAllShaders();
            //initBuffers();

        }

        this.drawScene = function() {

            /*canvas2d_element = document.getElementById("canvas-transfer");
            canvas2d = canvas2d_element.getContext("2d");
            canvas2d.strokeStyle="purple"
            canvas2d.strokeStyle="purple"
            canvas2d.fillStyle="#000000";
            canvas2d.fillRect(0, 0, 512, 30);
            canvas2d.drawImage(colormap_image, 0, 70-1-colormap_index, 1024, 1, data_min*512, 0, (data_max - data_min)*512, 30)
            var data_scale = 1./(data_max - data_min);
            clamp = function(x, xmin, xmax) {
                return (x < xmin ? xmin : (x > xmax ? xmax : x));
            }
            sign = function(x) {
                return Math.abs(x) / x;
            }
            for(var j = 0; j < 4; j++) {
                canvas2d.beginPath();
                canvas2d.moveTo(data_min*512, 0);
                for(var x_index = 0; x_index < 512; x_index++) {
                    var x = x_index/511;
                    var data_value = (x - data_min) * data_scale;//, 0., 1.);
                    var volume_level_value = (volume_level[j] - data_min) * data_scale;//, 0., 1.);
                    var chi = (data_value-volume_level[j])/volume_width[j];
                    var chisq = Math.pow(chi, 2.);
                    var intensity = Math.exp(-chisq);
                    var y = 30-intensity*(Math.log(opacity[j])/Math.log(10)+5)/5. * 30 * sign(data_value) * sign(1.-data_value);
                    if(x_index == 0) {
                        canvas2d.moveTo(x_index, y);
                    } else {
                        canvas2d.lineTo(x_index, y);
                    }
                    //vec4 color_sample = texture2D(colormap, vec2(clamp((level+2.)/2., 0., 1.), colormap_index_scaled));
                    //intensity = clamp(intensity, 0., 1.);
                    //float distance_norm = clamp(((-chi/0.5)+1.)/2., 0., 1.);
                    //color_index = 0.9;
                    //vec4 color_sample = texture2D(colormap, vec2(1.-volume_level[j], colormap_index_scaled));
                    //vec4 color_sample = texture2D(colormap, vec2(data_value, colormap_index_scaled));
                }
                canvas2d.stroke()
            }*/


            gl.bindFramebuffer(gl.FRAMEBUFFER, frame_buffer);
            gl.cullFace(gl.BACK);

            gl.enable(gl.CULL_FACE);

            gl.cullFace(gl.BACK);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture_frame_buffer_front, 0);
            this.drawCube(shader_cube);

            gl.cullFace(gl.FRONT);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture_frame_buffer_back, 0);
            this.drawCube(shader_cube);


            gl.cullFace(gl.BACK);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture_frame_buffer_volume, 0);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture_frame_buffer_back);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, texture_frame_buffer_front);
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, texture_volume);
            gl.activeTexture(gl.TEXTURE3);
            gl.bindTexture(gl.TEXTURE_2D, texture_colormaps);
            gl.activeTexture(gl.TEXTURE4);
            gl.bindTexture(gl.TEXTURE_2D, texture_transfer_function);
            gl.activeTexture(gl.TEXTURE0);

            gl.useProgram(shader_volume_rendering);
            gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "back"),  0);
            gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "front"),   1);
            gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "volume"), 2);
            gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "colormap"), 3);
            gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "transfer_function"),  4);
            gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "colormap_index"), options.colormap_index);
            gl.uniform1f(gl.getUniformLocation(shader_volume_rendering, "brightness"), options.brightness);
            gl.uniform1f(gl.getUniformLocation(shader_volume_rendering, "data_min"), options.data_min);
            gl.uniform1f(gl.getUniformLocation(shader_volume_rendering, "data_max"), options.data_max);
            log(options)

            gl.uniform1fv(gl.getUniformLocation(shader_volume_rendering, "opacity"),  options.opacity);
            gl.uniform1fv(gl.getUniformLocation(shader_volume_rendering, "volume_level"), options.volume_level);
            gl.uniform1fv(gl.getUniformLocation(shader_volume_rendering, "volume_width"), options.volume_width);

            this.drawCube(shader_volume_rendering);

            gl.cullFace(gl.BACK);
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            this.drawTexture();
        }
        this.drawTexture = function () {
            gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
            gl.clearColor(0.0, 0.0, 1.0, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.useProgram(shader_texture);

            //mat4.perspective(45, gl.viewportWidth / gl.viewportHeight, 0.1, 100.0, pMatrix);
            var size = 1.0;
            mat4.ortho(-size, size, -size, size, -100, 100, pMatrix)

            mat4.identity(mvMatrix);

            mat4.translate(mvMatrix, [0, 0.0, -7.0]);
            gl.bindBuffer(gl.ARRAY_BUFFER, squareVertexPositionBuffer);
            gl.vertexAttribPointer(shader_cube.vertexPositionAttribute, squareVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
            this.setMatrixUniforms(shader_texture);

            gl.bindTexture(gl.TEXTURE_2D, texture_show);
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, squareVertexPositionBuffer.numItems);
        }

        this.drawCube = function (program) {
            gl.viewport(0, 0, frame_buffer.width, frame_buffer.height);
            gl.clearColor(0.0, 0.0, 0.0, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

            gl.useProgram(program);
            mat4.perspective(15, gl.viewportWidth / gl.viewportHeight, 0.1, 100.0, pMatrix);
            var size = 0.6;///Math.sqrt(2);
            mat4.ortho(-size, size, -size, size, -100, 100, pMatrix)

            mat4.identity(mvMatrix);

            mat4.translate(mvMatrix, [-1.5, 0.0, -7.0]);
            gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);
            gl.vertexAttribPointer(shader_cube.vertexPositionAttribute, triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
            this.setMatrixUniforms(program);
            gl.drawArrays(gl.TRIANGLES, 0, triangleVertexPositionBuffer.numItems);


            /*mat4.translate(mvMatrix, [3.0, 0.0, 0.0]);
            gl.bindBuffer(gl.ARRAY_BUFFER, squareVertexPositionBuffer);
            gl.vertexAttribPointer(shader_cube.vertexPositionAttribute, squareVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
            setMatrixUniforms();
            gl.drawArrays(gl.TRIANGLE_STRIP, 0, squareVertexPositionBuffer.numItems);
            */
            var scale = vec3.create();
            vec3.set(scale, 0.8,0.8,0.8)

            mat4.identity(mvMatrix);
            //mat4.scale(mvMatrix, scale);
            mat4.translate(mvMatrix, [-0.0, 0.0, -7.0]);
            mat4.rotateY(mvMatrix, options.angle1);
            mat4.rotateX(mvMatrix, options.angle2);
            mat4.translate(mvMatrix, [-0.5, -0.5, -0.5]);
            gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer);
            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeIndexBuffer);
            gl.vertexAttribPointer(shader_cube.vertexPositionAttribute, cubeVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
            this.setMatrixUniforms(program);
            //gl.drawArrays(gl.TRIANGLES, cubeIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
            gl.drawElements(gl.TRIANGLES, cubeIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
            //gl.drawElements(gl.LINES, cubeIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);

        }
        this.setMatrixUniforms = function (program) {
            gl.uniformMatrix4fv(program.pMatrixUniform, false, pMatrix);
            gl.uniformMatrix4fv(program.mvMatrixUniform, false, mvMatrix);
        }


    	var update_counter = 0;
        this.updateScene = function() {
            (function(local_update_counter) {
                setTimeout(function(){
                    //console.log("update: " +update_counter +" / " +local_update_counter);
                    if(local_update_counter == update_counter) {
                        shader_volume_rendering = shader_volume_rendering_final;
                        plugin.real_updateScene()
                    }
                }, 300);
            })(++update_counter);
            shader_volume_rendering = shader_volume_rendering_updates;
            plugin.real_updateScene()

        }
        this.real_updateScene = function() {
            //console.log(localStorage)
            //localStorage.save("last_settings", settings)
            gl.clearColor(0.0, 1.0, 0.0, 1.0);
            gl.enable(gl.DEPTH_TEST);

            this.drawScene();
        }


		gl.clearColor(0.0, 0.0, 0.0, 1.0);
		gl.enable(gl.DEPTH_TEST);
		this.drawScene();

        console.log(transfer_function_array)

        var mouse_x = 0;
        var mouse_y = 0;
        var mouse_down = false;

        var canvas_ontouchmove = function( event ) {
            var msg = "Handler for .touchmove() called at ";

            var touch = event.originalEvent.touches[0] || event.originalEvent.changedTouches[0];
            var elm = $(this).offset();
            var x = touch.pageX - elm.left;
            var y = touch.pageY - elm.top;
            //$( "#log" ).append( "<div>" + x +"," + y + "</div>" );
            if(x < $(this).width() && x > 0){
                if(y < $(this).height() && y > 0){
                    //CODE GOES HERE
                    //console.log(touch.pageY+' '+touch.pageX);
                    msg += x + ", " + y;
                    var dx = x - mouse_x;
                    var dy = y - mouse_y;
                    var speed = 0.01;
                    options.angle1 += dx * speed;
                    options.angle2 += dy * speed;
                    //var msg = "change angle" + angle1 + ", " + angle2;
                    plugin.updateScene();
                    mouse_x = x;
                    mouse_y = y;
                    //$( "#log" ).append( "<div>" + msg + "</div>" );


                    event.preventDefault();
                }
            }
        }
        var canvas_onmousemove = function( event ) {
            var msg = "Handler for .mousemove() called at ";
            msg += event.pageX + ", " + event.pageY;
            var elm = $(this).offset();
            if(elm) {
                var x = event.pageX - elm.left;
                var y = event.pageY - elm.top;
                //$( "#log" ).append( "<div>" + x +"," + y + "</div>" );
                if(x < $(this).width() && x > 0){
                    if(y < $(this).height() && y > 0){
                        if(mouse_down) {
                            var dx = event.pageX - mouse_x;
                            var dy = event.pageY - mouse_y;
                            var speed = 0.01;
                            options.angle1 += dx * speed;
                            options.angle2 += dy * speed;
                           // var msg = "change angle" + angle1 + ", " + angle2;
                            plugin.updateScene();
                        }
                    }
                }
                mouse_x = event.pageX;
                mouse_y = event.pageY;
                $( "#log" ).append( "<div>" + msg + "</div>" );
            }
            event.preventDefault();
        }
        var canvas_onmousedown = function(event){
            console.log("down")
            var elm = $(this).offset();
            console.log(elm)
            var msg = "Handler for .mousedown() called at ";
            msg += event.pageX + ", " + event.pageY;
            //$( "#log" ).append( "<div>" + msg + "</div>" );
            //event.preventDefault();
            mouse_down = true;
        }
        var canvas_onmouseup = function(event){
            console.log("up")
            var msg = "Handler for .mouseup() called at ";
            msg += event.pageX + ", " + event.pageY;
            //$( "#log" ).append( "<div>" + msg + "</div>" );
            event.preventDefault();
            mouse_down = false;
        }

		$(canvas).mousedown(canvas_onmousedown);
		$(canvas).on("touchmove", canvas_ontouchmove);
		$(canvas).mouseup(canvas_onmouseup);
		$(canvas).mousemove(canvas_onmousemove);
        //plugin.init()
    }
    $.fn.vr = function(options) {
        var settings = $.extend({
            // These are the defaults.
            color: "red",
            backgroundColor: "black",
            frame_buffer_width: 256,
            frame_buffer_height: 256,
            angle1: 0.2,
            angle2: 0.2,
            colormap_index:0,
            data_min:0.,
            data_max:0.1,
            opacity:[0.04, 0.01, 0.1],
            brightness: 2.,
            volume_level: [0.1, 0.5, 0.75],
            volume_width: [0.1, 0.1, 0.2],
            cube:"cube.png",
            colormap:"colormap.png"
        }, options );
        log(settings)
//        #volume_level: [178/255., 85/255., 31/255., 250],
        log(this)
        return this.each(function() {
            var plugin = new $.vr(this, settings)
            log("return plugin")
            $(this).data("vr", plugin)
            return plugin
        });
    }
    function log(obj) {
        if ( window.console && window.console.log ) {
            window.console.log(obj );
        }
    };

}(jQuery));
