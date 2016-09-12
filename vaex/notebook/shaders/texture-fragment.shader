precision mediump float;
varying vec4 vertex_color;
varying vec2 vTextureCoord;
uniform sampler2D uSampler; // default is 0, so we don't have to set it

void main(void) {
vec4 textureColor = texture2D(uSampler, vec2(vTextureCoord.s, vTextureCoord.t));
    gl_FragColor = vec4(vTextureCoord.s, vTextureCoord.t, 0, 1);
gl_FragColor = textureColor;
}
