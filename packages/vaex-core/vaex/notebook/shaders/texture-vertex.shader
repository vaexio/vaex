attribute vec3 aVertexPosition;
attribute vec2 aTextureCoord;

uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;

varying vec4 vertex_color;
varying vec2 vTextureCoord;

void main(void) {
    gl_Position = uPMatrix * uMVMatrix * vec4(aVertexPosition, 1.0);
//vertex_color = vec4(aVertexPosition, 1);
vTextureCoord = vec2(aVertexPosition.x/2.+0.5, aVertexPosition.y/2. + 0.5);
}
