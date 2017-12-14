precision highp float;
varying vec4 vertex_color;
uniform sampler2D front;
uniform sampler2D back;
uniform sampler2D volume;
uniform sampler2D colormap;
uniform int colormap_index;
uniform int surfaces;
uniform float opacity[4];
uniform float volume_level[4];
uniform float volume_width[4];
uniform float brightness;
uniform float data_min;
uniform float data_max;

uniform sampler2D transfer_function;

//uniform float color_index;

vec2 computeSliceOffset(float slice, float slicesPerRow, vec2 sliceSize) {
return sliceSize * vec2(mod(slice, slicesPerRow),
            floor(slice / slicesPerRow));
}
vec4 sampleAs3DTexture(sampler2D tex, vec3 texCoord, float size, float numRows, float slicesPerRow) {
  //float slice   = texCoord.z*0. + 1.*size*(size-1.)/size ;
  float slice   = texCoord.z*size*(size-1.)/size ;
  float sliceZ  = floor(slice);                         // slice we need
  float zOffset = fract(slice);                         // dist between slices
  //sliceZ = 64.;

  vec2 sliceSize = vec2((1.0-1./2048.) / slicesPerRow,             // u space of 1 slice
              (1.0-1./1024.) / numRows);                 // v space of 1 slice

  vec2 slice0Offset = computeSliceOffset(sliceZ, slicesPerRow, sliceSize);
  vec2 slice1Offset = computeSliceOffset(sliceZ + 1.0, slicesPerRow, sliceSize);

  vec2 slicePixelSize = sliceSize / size;               // space of 1 pixel
  vec2 sliceInnerSize = slicePixelSize * (size - 1.0);  // space of size pixels

  vec2 coord = vec2(texCoord.x, texCoord.y);
  vec2 uv = slicePixelSize * 0.5 + coord * sliceInnerSize;
  vec4 slice0Color = texture2D(tex, slice0Offset + uv);
  //vec2 uv1 = slice0Offset + uv.xy;
  //vec4 slice0Color = texture2D(tex, vec2(uv1.x, uv1.y));
  vec4 slice1Color = texture2D(tex, slice1Offset + uv);
  return mix(slice0Color, slice1Color, zOffset);
  ///return slice0Color;
}

void main(void) {
  const int steps = NR_OF_STEPS;

  vec2 pixel = vec2(gl_FragCoord.x/256., gl_FragCoord.y/256.);
  //vec4 textureColor = texture2D(volume, vec2(pixel.x * width + x_index, pixel.y*height + y_index));
  vec3 ray_begin = vertex_color.xyz;//texture2D(front, pixel).rgb;
  vec3 ray_end = texture2D(back, pixel).rgb;
  vec3 ray_direction = ray_end - ray_begin;
  vec3 ray_delta = ray_direction * (1./float(steps));
  vec3 ray_pos = ray_begin;
  vec4 color = vec4(0, 0, 0, 0);
  float alpha_total = 0.;
  float colormap_index_scaled = 0.5/70. + float(colormap_index)/70.;
  float color_index;
  float data_scale = 1./(data_max - data_min);
  for(int i = 0; i < steps; i++) {
    vec3 pos = ray_pos;
    //pos = vec3(pos.xy, 1./128. * mod1); //ray_pos.z * 0.9 + 0.05);
    //pos = vec3(1./128. * mod1, pos.yz); //ray_pos.z * 0.9 + 0.05);
    //pos = vec3(pos.x, pos.yz); //ray_pos.z * 0.9 + 0.05);
    //pos = vec3(ray_pos.xy, 1./128. * mod1); //ray_pos.z * 0.9 + 0.05);
    //pos = vec3(ray_pos.x, ray_pos.y, 1./128. * mod1); //ray_pos.z * 0.9 + 0.05);
    //vec4 color_sample = sampleAs3DTexture(volume, pos, 128., 8., 16.);
    vec4 sample = sampleAs3DTexture(volume, pos, 128., 8., 16.);
    //for(int j = 0; j < 3; j++) {
      //float bla = length(sample)/sqrt(3.);
      float data_value = (sample.a - data_min) * data_scale;
      //float volume_level_value = (volume_level[j] - data_min) * data_scale;;
      //float chi = (data_value-volume_level[j])/volume_width[j];
      //float chisq = pow(chi, 2.);
      //float intensity = exp(-chisq);
      //vec4 color_sample = texture2D(colormap, vec2(clamp((level+2.)/2., 0., 1.), colormap_index_scaled));
      //intensity = clamp(intensity, 0., 1.);
      //float distance_norm = clamp(((-chi/0.5)+1.)/2., 0., 1.);
      //color_index = 0.9;
      //vec4 color_sample = texture2D(colormap, vec2(1.-volume_level[j], colormap_index_scaled));
      vec4 color_sample = texture2D(colormap, vec2(data_value, colormap_index_scaled));

      float intensity = texture2D(transfer_function, vec2(data_value, 0.5)).a;
      //color_sample = texture2D(transfer_function, data_value);
      //vec4 color_sample = texture2D(colormap, vec2(sample.a, colormap_index_scaled));
      //color_sample = texture2D(volume, ray_pos.yz);
      //float alpha_sample = opacity*intensity;//1./128.* length(color_sample) * 100.;
      float alpha_sample = intensity * sign(data_value) * sign(1.-data_value) * 100. / float(steps);//clamp(1.-chisq, 0., 1.) * 0.5;//1./128.* length(color_sample) * 100.;
      alpha_sample = clamp(alpha_sample, 0., 1.);
      color = color + (1.0 - alpha_total) * color_sample * alpha_sample;
      if(alpha_total >= 1.)
        break;
      alpha_total = clamp(alpha_total + alpha_sample, 0., 1.);
    //}
    ray_pos += ray_delta;
  }
  gl_FragColor = vec4(color.rgb, 1) * brightness;
  //float tintensity = texture2D(transfer_function, vec2(pixel.x / 1., 0.5)).a;
  //gl_FragColor = vec4(0, tintensity, 0., 1.);
  //gl_FragColor = vec4(ray_e, 1);
  }
