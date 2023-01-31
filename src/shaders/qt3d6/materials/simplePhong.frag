#version 150 core

in vec2 texCoord;
in vec3 normal;
in vec3 color;
in vec3 viewPos;
//in vec3 lightA;
//in vec3 lightB;
in vec3 worldNormal;
in vec3 worldPosition;
out vec4 fragColor;

vec3 lightA = vec3(0,0,0);
vec3 lightB = vec3(-800,2000,-3500);

void main()
{
	// ambient
       vec3 ambient = vec3(0.1, 0.1, 0.1);
       
       // diffuse
	vec3 normalN = normalize(worldNormal);


   	vec3 lightDirA = normalize(lightA - worldPosition);
   	vec3 lightDirB = normalize(lightB - worldPosition);


	vec3 diffuseA = 0.7 * vec3(1,1,1) * max(dot(normalN,lightDirA), 0.0);
	vec3 diffuseB = 0.6 * vec3(1,1,1) * max(dot(normalN,lightDirB), 0.0);
	
	vec3 diffuse = clamp(diffuseA + diffuseB, 0.0, 1.0);
	
	// specular	
	vec3 viewDir = normalize(-viewPos) ;
	float shininess = 64;   
	vec3 reflectDirA = normalize(reflect(lightDirA, normalN));

	vec3 reflectDirB = normalize(reflect(lightDirB, normalN)); 	 	

	vec3 specA = vec3(1,1,1) * pow(max(dot(viewDir, reflectDirA), 0.0), shininess);
	vec3 specB = vec3(1,1,1) * pow(max(dot(viewDir, reflectDirB), 0.0), shininess);
	
	vec3 spec = clamp(specA + specB, 0.0, 1.0);	
	
	vec3 result = ambient * color + diffuse * color + spec;

	fragColor = vec4(result, 1.0);
}
