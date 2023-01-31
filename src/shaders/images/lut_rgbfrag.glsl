#version 330 core

// vertices datas
in vec2 v_textureCoordinates;

// uniforms
uniform sampler1D color_map;
uniform sampler2D f_tileTexture; // tile texture
uniform float f_opacity = 0.25; // tile opacity
uniform float f_rangeMin = 0.0; // tile min
uniform float f_rangeRatio = 1.0; // tile min

uniform bool f_noHasDataValue = false; //has no data
uniform float f_noDataValue = 1.0; //NoDataValue

out vec4 f_fragColor; // shader output color

void main()
{
    vec4 color = texture(f_tileTexture, v_textureCoordinates.st);
    f_fragColor = vec4(color.x,color.y,color.z,f_opacity);
}