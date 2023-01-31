#version 150 core

uniform vec3 lightPosition;
uniform vec3 colorObj;
in vec3 worldPosition;
in vec3 worldNormal;

out vec4 fragColor;

float kd = 0.7;
float ka = 0.1;
//float att = 0.00;

void main()
{
	vec3 _worldNormal = worldNormal;
	//vec3 color1 = vec3(1,0,0);
	if(gl_FrontFacing)
	{
	 _worldNormal = worldNormal*-1.0;
	
	}
	_worldNormal = normalize(_worldNormal);
	
	vec3 L = normalize(lightPosition-worldPosition);
//	vec3 L = normalize(lightPosition);
	
	float LdotN= max(0.0,dot(L,_worldNormal));
	//float distance = length(lightPosition - worldPosition);
	//float attenuation = 1.0 / (1.0 + att * pow(distance, 2));
	vec3 diffus = ka *colorObj + kd* colorObj* LdotN;
	fragColor = vec4(diffus,1.0);
//	gl_FragDepth = gl_FragCoord.z;
}