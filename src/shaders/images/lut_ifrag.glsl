#version 330 core

// vertices datas
in vec2 v_textureCoordinates;

// uniforms
uniform sampler1D color_map;
uniform isampler2D f_tileTexture; // tile texture
uniform float f_opacity = 0.25; // tile opacity
uniform float f_rangeMin = 0.0; // tile min
uniform float f_rangeRatio = 1.0; // tile min

uniform bool f_noHasDataValue = false; //has no data
uniform float f_noDataValue = 1.0; //NoDataValue
uniform isampler2D mask_Texture;

out vec4 f_fragColor; // shader output color

float texture_bilinear(in isampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize)
{
    float tl = texture(t, uv).r;
    float tr = texture(t, uv + vec2(texelSize.x, 0.0)).r;
    float bl = texture(t, uv + vec2(0.0, texelSize.y)).r;
    float br = texture(t, uv + vec2(texelSize.x, texelSize.y)).r;
      
    
    vec2 f = fract( uv * textureSize );
    float tA = mix( tl, tr, f.x );
    float tB = mix( bl, br, f.x );
    return mix( tA, tB, f.y );
}

void main()
{
    //float origval = texture(f_tileTexture, v_textureCoordinates.st).r;     
   
    ivec2 texSize = textureSize(f_tileTexture, 0);
    vec2 texelSize = vec2(1.0/texSize);
    float origval = texture_bilinear(f_tileTexture, v_textureCoordinates.st, texSize, texelSize);


    float val = (origval - f_rangeMin) * f_rangeRatio;
    
    if(val>1.0)
    	val=1.0;
    if(val<0.0)
    	val=0.0;
    
    vec4 color=texture(color_map, val);	

    float opacity=min(color.a,f_opacity);
    if(f_noHasDataValue && f_noDataValue==origval)
    {
        opacity=0.0f;
    }
	
    float  mask = texture(mask_Texture, v_textureCoordinates.st).r/255.0;
    f_fragColor = vec4(color.r,color.g,color.b,opacity*mask);
}
