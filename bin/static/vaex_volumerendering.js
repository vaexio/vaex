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


    function getShader(gl, id, replacements) {
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


    var shaderProgram;
	var shader_texture;
	var shader_volume_rendering_poor;
	var shader_volume_rendering_fast;
	var shader_volume_rendering_best;
	var shader_volume_rendering;

    function initShaders(name, replacements) {
		var fragmentShader = getShader(gl, "shader-fragment-"+name, replacements);
		var vertexShader = getShader(gl, "shader-vertex-"+name, replacements);
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
	function initAllShaders() {
        shaderProgram = initShaders("cube");
        gl.useProgram(shaderProgram);
        shader_texture = initShaders("texture");
        shader_volume_rendering_best = initShaders("volume-rendering", {NR_OF_STEPS:300});
        shader_volume_rendering_fast = initShaders("volume-rendering", {NR_OF_STEPS:80});
        shader_volume_rendering_poor = initShaders("volume-rendering", {NR_OF_STEPS:40});
        shader_volume_rendering = shader_volume_rendering_fast;
		shader_volume_rendering_updates = shader_volume_rendering_poor;
		shader_volume_rendering_final = shader_volume_rendering_best;
        //gl.useProgram(shaderProgram);
    }


    var mvMatrix = mat4.create();
    var pMatrix = mat4.create();

    function setMatrixUniforms(program) {
        gl.uniformMatrix4fv(program.pMatrixUniform, false, pMatrix);
        gl.uniformMatrix4fv(program.mvMatrixUniform, false, mvMatrix);
    }



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
	var texture_colormaps;
	
	var texture_volume;
	var volume_size = 128;
	var colormap_image;
	
	function initVolumeTexture() {
		texture_volume = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture_volume);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);		
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, volume_size, volume_size, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
		loadTexture(texture_volume, gl.RGBA, "cube.png");
		//var ext = gl.getExtension("OES_texture_float")
		//console.log("ext:" + ext.FLOAT);
	}
	
	function initColormapTexture() {
		texture_colormaps = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture_colormaps);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);		
		gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, volume_size, volume_size, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
		colormap_image = loadTexture(texture_colormaps, gl.RGB, "colormap.png");
		//var ext = gl.getExtension("OES_texture_float")
		//console.log("ext:" + ext.FLOAT);
	}

	function loadTexture(texture, format, url) {
		var textureImage = new Image();
		textureImage.onload = function() { 
			gl.bindTexture(gl.TEXTURE_2D, texture);
			gl.texImage2D(gl.TEXTURE_2D, 0, format, format, gl.UNSIGNED_BYTE, textureImage);
			//alert("loaded: " +volumeImage.src + " " +gl.getError() + ":" +volumeImage);
			updateScene();
		}
		textureImage.src = url;
		return textureImage;
		
	}
	
	
	
    function initBuffers() {
		initVolumeTexture()
		initColormapTexture();
		frame_buffer = gl.createFramebuffer();
		gl.bindFramebuffer(gl.FRAMEBUFFER, frame_buffer);
		frame_buffer.width = 256;
		frame_buffer.height = 256;


		texture_frame_buffer_volume = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture_frame_buffer_volume);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, frame_buffer.width, frame_buffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);


		texture_frame_buffer_back = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture_frame_buffer_back);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
		//gl.generateMipmap(gl.TEXTURE_2D);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, frame_buffer.width, frame_buffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

		texture_frame_buffer_front = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, texture_frame_buffer_front);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
		//gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
		//gl.generateMipmap(gl.TEXTURE_2D);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, frame_buffer.width, frame_buffer.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);

		texture_show = texture_frame_buffer_volume;//texture_frame_buffer_back;

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
        indices = [
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

    }


    function drawScene() {
		
		canvas2d_element = document.getElementById("canvas-transfer");
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
		}
		
		
		gl.bindFramebuffer(gl.FRAMEBUFFER, frame_buffer);
		gl.cullFace(gl.BACK);

		gl.enable(gl.CULL_FACE);

		gl.cullFace(gl.BACK);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture_frame_buffer_front, 0);
		drawCube(shaderProgram);

		gl.cullFace(gl.FRONT);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture_frame_buffer_back, 0);
		drawCube(shaderProgram);

		
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
		gl.activeTexture(gl.TEXTURE0);
		
        gl.useProgram(shader_volume_rendering);
		gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "back"),  0);
		gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "front"),   1);
		gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "volume"), 2);
		gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "colormap"), 3);
		gl.uniform1i(gl.getUniformLocation(shader_volume_rendering, "colormap_index"), colormap_index);
		gl.uniform1f(gl.getUniformLocation(shader_volume_rendering, "brightness"), brightness);
		gl.uniform1f(gl.getUniformLocation(shader_volume_rendering, "data_min"), data_min);
		gl.uniform1f(gl.getUniformLocation(shader_volume_rendering, "data_max"), data_max);
		
		gl.uniform1fv(gl.getUniformLocation(shader_volume_rendering, "opacity"),  opacity);
		gl.uniform1fv(gl.getUniformLocation(shader_volume_rendering, "volume_level"), volume_level);
		gl.uniform1fv(gl.getUniformLocation(shader_volume_rendering, "volume_width"), volume_width);
		
		drawCube(shader_volume_rendering);
		
		gl.cullFace(gl.BACK);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		drawTexture();
	}
	function drawTexture() {
		gl.viewport(0, 0, gl.viewportWidth, gl.viewportHeight);
		gl.clearColor(0.0, 1.0, 0.0, 1.0);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		gl.useProgram(shader_texture);

		//mat4.perspective(45, gl.viewportWidth / gl.viewportHeight, 0.1, 100.0, pMatrix);
		var size = 1.0;
		mat4.ortho(-size, size, -size, size, -100, 100, pMatrix)

		mat4.identity(mvMatrix);

		mat4.translate(mvMatrix, [0, 0.0, -7.0]);
		gl.bindBuffer(gl.ARRAY_BUFFER, squareVertexPositionBuffer);
		gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, squareVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
		setMatrixUniforms(shader_texture);

		gl.bindTexture(gl.TEXTURE_2D, texture_show);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, squareVertexPositionBuffer.numItems);
	}

    function drawCube(program) {
        gl.viewport(0, 0, frame_buffer.width, frame_buffer.height);
		gl.clearColor(1.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.useProgram(program);
        mat4.perspective(15, gl.viewportWidth / gl.viewportHeight, 0.1, 100.0, pMatrix);
		var size = 0.6;///Math.sqrt(2);
		mat4.ortho(-size, size, -size, size, -100, 100, pMatrix)

        mat4.identity(mvMatrix);

        mat4.translate(mvMatrix, [-1.5, 0.0, -7.0]);
        gl.bindBuffer(gl.ARRAY_BUFFER, triangleVertexPositionBuffer);
        gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, triangleVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
        setMatrixUniforms(program);
        gl.drawArrays(gl.TRIANGLES, 0, triangleVertexPositionBuffer.numItems);


        /*mat4.translate(mvMatrix, [3.0, 0.0, 0.0]);
        gl.bindBuffer(gl.ARRAY_BUFFER, squareVertexPositionBuffer);
        gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, squareVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
        setMatrixUniforms();
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, squareVertexPositionBuffer.numItems);
		*/
		var scale = vec3.create();
		vec3.set(scale, 0.8,0.8,0.8)

        mat4.identity(mvMatrix);
		//mat4.scale(mvMatrix, scale);
		mat4.translate(mvMatrix, [-0.0, 0.0, -7.0]); 
        mat4.rotateY(mvMatrix, angle1);
        mat4.rotateX(mvMatrix, angle2);
		mat4.translate(mvMatrix, [-0.5, -0.5, -0.5]); 
        gl.bindBuffer(gl.ARRAY_BUFFER, cubeVertexPositionBuffer);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, cubeIndexBuffer);
        gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute, cubeVertexPositionBuffer.itemSize, gl.FLOAT, false, 0, 0);
        setMatrixUniforms(program);
        //gl.drawArrays(gl.TRIANGLES, cubeIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
        gl.drawElements(gl.TRIANGLES, cubeIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);
        //gl.drawElements(gl.LINES, cubeIndexBuffer.numItems, gl.UNSIGNED_SHORT, 0);

    }

