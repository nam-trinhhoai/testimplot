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
	
	vec3 L = normalize(lightPosition-worldPosition);
	
	float LdotN= max(0.0,dot(L,_worldNormal));
	//float distance = length(lightPosition - worldPosition);
	//float attenuation = 1.0 / (1.0 + att * pow(distance, 2));
	vec3 diffus = ka *colorObj + kd* colorObj* LdotN;
	fragColor = vec4(diffus,1.0);
//	gl_FragDepth = gl_FragCoord.z;
}

/*
#version 430 core



in vec3 position;

in vec3 normal;
uniform vec3 colorObj;

out vec4 fragColor;

void main()
{
	vec3 posLight0 = vec3(0,-1000,0);
	vec3 dirLight0 = position - posLight0;


//	vec3 dirLight1= vec3(-0.1,-1,0.0);
//	dirLight1 = normalize(dirLight1);
	
	//vec3 dirLight2= vec3(0.1,1,0.05);
//dirLight2 = normalize(dirLight2);
	vec3 ambient = colorObj*0.0;
	vec3 diffuse = colorObj*0.0;//0.5
	vec3 normalN = normalize(normal);
//	vec3 color =ambient+abs( dot(normalN,dirLight1))*diffuse+ abs(dot(normalN,dirLight2))*diffuse;
	vec3 color =(dot(normalN,dirLight0))*diffuse;
    fragColor = vec4(color,1);
}
*/

/*#version 150 core



in vec3 position;
in vec3 normal;


out vec4 fragColor;

void main()
{
  
    vec3 color = dot(normal,vec3(0,0,1))*vec3(0,1,0);
    fragColor = vec4(1.0,1.0,0.0,1.0);
  //  gl_FragDepth = gl_FragCoord.z;
}


*/


/*
#version 430 core

//uniform sampler2D diffuseTexture;
//uniform float opacity;

//uniform sampler1D samImage;
//uniform sampler2D samImage;


in vec3 position;
in vec3 col;
in vec3 norm;
//in vec2 texCoord;

out vec4 fragColor;

void main()
{
   // vec3 texval = position;

   // vec3 dx = dFdx(position);
    //vec3 dy = dFdy(position);
     vec3 normal=norm;
    //vec3 normal=normalize(cross(dx,dy));

   // vec3 color = dot(normal,vec3(0,0,1))*vec3(0,1,0);
    fragColor = vec4(1.0,1.0,0.0,1.0);
}

*/