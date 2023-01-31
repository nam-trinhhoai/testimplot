#version 330 core

// vertices datas
in vec2 v_textureCoordinates;

// uniforms
uniform sampler1D color_map;
uniform isampler2D f_tileTexture; // tile texture
uniform float f_opacity = 0.25; // tile opacity
uniform float f_rangeMin = 0.0; // tile min
uniform float f_rangeRatio = 0.1; // tile min

uniform bool f_noHasDataValue = false; //has no data
uniform float f_noDataValue = 1.0; //NoDataValue

out vec4 f_fragColor; // shader output color

void main()
{
	float origval = texture(f_tileTexture, v_textureCoordinates.st).r;
    float val = (origval - f_rangeMin) * f_rangeRatio;
    
    if(val>1.0)
    	val=1.0;
    if(val<0.0)
    	val=0.0;
    
    vec4 color=texture(color_map, val);	
    float opacity=min(color.a,f_opacity);
     	
    //Transparent if no data
	if(f_noHasDataValue && f_noDataValue==origval)
	{
		opacity=0.0f;
	}
   
    f_fragColor = vec4(color.r,color.g,color.b,opacity);
}
