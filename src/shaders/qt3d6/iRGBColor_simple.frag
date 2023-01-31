#version 150 core

uniform isampler2D redMap;
uniform isampler2D greenMap;
uniform isampler2D blueMap;

//uniform isampler2D surfaceMap;

uniform float opacity = 0.25; // tile opacity

uniform vec2 redRange;
uniform vec2 greenRange;
uniform vec2 blueRange;

in vec3 position;
in vec3 normal;
in vec2 texCoord;
in float potentialHole;



out vec4 fragColor;

float clip(float val)
{
 	if(val>1.0)
		return 1.0;
	if(val<0.0)
		return 0.0;
	return val;
}

void main()
{	
if(potentialHole > 0.0) {
 	
		discard;
	}
	
    float redval = clip((texture(redMap, texCoord.st).r - redRange.x) *redRange.y);
    float greenval = clip((texture(greenMap, texCoord.st).r - greenRange.x) *greenRange.y);
    float blueval = clip((texture(blueMap, texCoord.st).r - blueRange.x) *blueRange.y);

    vec3 normalH= normalize(normal);
    fragColor = vec4(redval,greenval,blueval,opacity);

  //  gl_FragDepth = gl_FragCoord.z;

}

