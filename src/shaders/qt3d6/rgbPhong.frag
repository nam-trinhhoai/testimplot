#version 150 core

uniform isampler2D rgbMap;

uniform vec2 redRange;
uniform vec2 greenRange;
uniform vec2 blueRange;
uniform float opacity;
uniform float heightThreshold;

uniform bool minValueActivated = false;
uniform float minValue = 0.0;

in vec3 position;
in vec3 normal;
in vec2 texCoord;
in float potentialHole;

out vec4 fragColor;

float texture_bilinear(in isampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize, float rangeX, float rangeY)
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

// Function from https://thebookofshaders.com/06/
vec3 rgb2hsb(in vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz),
                 vec4(c.gb, K.xy),
                 step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r),
                 vec4(c.r, p.yzx),
                 step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)),
                d / (q.x + e),
                q.x);
}

//  Function from Iñigo Quiles
//  https://www.shadertoy.com/view/MsS3Wc
vec3 hsb2rgb(in vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),
                             6.0)-3.0)-1.0,
                     0.0,
                     1.0 );
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z * mix(vec3(1.0), rgb, c.y);
}

void main()
{
	
	if(potentialHole > 0.0) {
 	
		discard;
	}
	ivec2 texSize = textureSize(rgbMap, 0);
	vec2 texelSize = vec2(1.0/texSize);

    //float redval = clamp((texture(redMap, texCoord.st).r - redRange.x) *redRange.y,0.0,1.0);  
	//float greenval = clamp((texture(greenMap, texCoord.st).r - greenRange.x) *greenRange.y,0.0,1.0);
	//float blueval = clamp((texture(blueMap, texCoord.st).r - blueRange.x) *blueRange.y,0.0,1.0);
	
	//float redval = texture_bilinear(redMap, texCoord.st, texSize, texelSize, redRange.x, redRange.y);
	//float greenval = texture_bilinear(greenMap, texCoord.st, texSize, texelSize, greenRange.x, greenRange.y);
	//float blueval = texture_bilinear(blueMap, texCoord.st, texSize, texelSize, blueRange.x, blueRange.y);
	
	vec3 color = texture(rgbMap, texCoord.st).rgb;
	color.r = clamp((color.r- redRange.x) * redRange.y,0.0,1.0);
	color.g = clamp((color.g- greenRange.x) * greenRange.y,0.0,1.0);
	color.b = clamp((color.b- blueRange.x) * blueRange.y,0.0,1.0);
	
   // vec3 color= vec3(redval,greenval,blueval);

	if (minValueActivated)
	{
		vec3 hsv = rgb2hsb(color);
		if (hsv.z<minValue) {
			hsv.z = minValue;
		}
		color = hsb2rgb(hsv);
	}

	vec3 normalN = normalize(normal);

   	vec3 lightDirA = normalize(vec3(0.0,-1.0,0.0));//normalize(lightPosition - position);
	vec3 diffuseA = 1.0* vec3(1,1,1) * max(dot(normalN,lightDirA), 0.0);
	
	vec3 ambient = vec3(0.1,0.1,0.1);
	vec3 diffuse = clamp(diffuseA , 0.0, 1.0);
	vec3 result = color *ambient  + diffuse*color ;
  	fragColor = vec4(result,opacity);
  	//gl_FragDepth = gl_FragCoord.z;
  }
