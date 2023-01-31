#version 150 core

uniform sampler1D colormap;
uniform isampler2D elementMap; // tile texture

uniform float opacity = 0.25; // tile opacity
uniform vec2 paletteRange;
uniform bool hover = false; 

in vec3 position;
in vec3 normal;
in vec2 texCoord;

out vec4 fragColor;

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
	if(hover && (texCoord.st.x<0.01 || texCoord.st.y<0.01 || texCoord.st.x>0.99|| texCoord.st.y>0.99))
	{
		fragColor = vec4(1.0,0.0,0.0,opacity);
		return;
	}
	//float origval = texture(elementMap, texCoord.st).r;
	ivec2 texSize = textureSize(elementMap, 0);
	vec2 texelSize = vec2(1.0/texSize);
	float origval = texture_bilinear(elementMap, texCoord.st, texSize, texelSize);
	
	
    	float val = (origval - paletteRange.x) *paletteRange.y;
    
    if(val>1.0)
    	val=1.0;
    if(val<0.0)
    	val=0.0;

    vec4 color=texture(colormap, val);	
    fragColor = vec4(color.r,color.g,color.b,min(color.a,opacity));
    gl_FragDepth = gl_FragCoord.z;
}

