#version 150 core

uniform sampler1D colormap;
uniform sampler2D elementMap; // tile texture

//uniform isampler2D surfaceMap;

uniform vec2 paletteRange;
uniform float opacity;
uniform float heightThreshold;

in vec3 position;
in vec3 normal;
in vec2 texCoord;
in float potentialHole;

out vec4 fragColor;

float texture_bilinear(in sampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize, float rangeX, float rangeY)
{
    float tl = clamp((texture(t, uv).r - rangeX) * rangeY,0.0,1.0);
    float tr = clamp((texture(t, uv + vec2(texelSize.x, 0.0)).r - rangeX) * rangeY,0.0,1.0);
    float bl = clamp((texture(t, uv + vec2(0.0, texelSize.y)).r - rangeX) * rangeY,0.0,1.0);
    float br = clamp((texture(t, uv + vec2(texelSize.x, texelSize.y)).r - rangeX) * rangeY,0.0,1.0);
      
    
    vec2 f = fract( uv * textureSize );
    float tA = mix( tl, tr, f.x );
    float tB = mix( bl, br, f.x );
    return mix( tA, tB, f.y );
}

void main()
{
	if(potentialHole > 0.0) {
 	
		discard;
	}
	ivec2 texSize = textureSize(elementMap, 0);
	vec2 texelSize = vec2(1.0/texSize);

	float val = texture_bilinear(elementMap, texCoord.st, texSize, texelSize, paletteRange.x, paletteRange.y);
	
	vec3 color=texture(colormap, val).xyz;
	vec3 normalN = normalize(normal);

   	vec3 lightDirA = normalize(vec3(0.0,-1.0,0.0));//normalize(lightPosition - position);
	vec3 diffuseA = 1.0* vec3(1,1,1) * max(dot(normalN,lightDirA), 0.0);
	
	vec3 ambient = vec3(0.1,0.1,0.1);
	vec3 diffuse = clamp(diffuseA , 0.0, 1.0);
	vec3 result = color *ambient  + diffuse*color ;
  fragColor = vec4(result,opacity);
  }
