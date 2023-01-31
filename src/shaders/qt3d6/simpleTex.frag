#version 150 core


uniform isampler2D redMap; // texture red
//uniform isampler2D greenMap; // texture red
//uniform isampler2D blueMap; // texture red

in vec3 position;
in vec3 normal;
in vec2 texCoord;

uniform vec2 redRange;
//uniform vec2 greenRange;
//uniform vec2 blueRange;

uniform bool hover = false; 
uniform float ratio = 1.0;
out vec4 fragColor;

void main()
{
	float borderH = 0.008;
	float borderW = borderH*ratio;
	if(hover && (texCoord.st.x<borderW || texCoord.st.y<borderH || texCoord.st.x>1.0-borderW|| texCoord.st.y>1.0-borderH))
	{
		fragColor = vec4(1.0,0.0,0.0,1.0);
		return;
	}
	float redval = clamp((texture(redMap, texCoord.st).r - redRange.x) *redRange.y,0.0,1.0);  
	//float greenval = clamp((texture(greenMap, texCoord.st).r - greenRange.x) *greenRange.y,0.0,1.0);  
	//float blueval = clamp((texture(blueMap, texCoord.st).r - blueRange.x) *blueRange.y,0.0,1.0);  
	//vec3 origval = texture(elementMap, texCoord.st).rgb;
    fragColor = vec4(redval,redval,redval,1.0);
   // gl_FragDepth = gl_FragCoord.z;

}