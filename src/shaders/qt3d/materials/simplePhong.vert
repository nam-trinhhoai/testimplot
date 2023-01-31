#version 150 core

in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec3 vertexColor;
in vec3 vertexNormal;

out vec3 worldPosition;
out vec3 worldNormal;
out vec3 viewPos;
out vec2 texCoord;
out vec3 normal;
out vec3 color;
//out vec3 lightA;
//out vec3 lightB;


uniform vec3 cameraPos;

uniform mat4 modelMatrix;
uniform mat3 modelViewNormal;
uniform mat3 modelNormalMatrix;
uniform mat4 worldMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelView;
uniform mat4 projectionMatrix;
uniform float aspectRatio;

uniform mat4 mvp;

void main()
{
	/*//transform to view space
	normal = normalize(modelViewNormal * vertexNormal);	
	//vec3 lightPosA = vec3(2500,200,500);
	//vec3 lightPosB = vec3(-800,2000,-3500);
	
	//lightA = (viewMatrix * vec4(lightPosA, 1.0)).xyz;
	//lightB = (viewMatrix * vec4(lightPosB, 1.0)).xyz;
	
	*/
		
	color = vertexColor;

	viewPos = vec3(modelView * vec4(vertexPosition, 1.0));
	gl_Position = mvp * vec4(vertexPosition, 1.0);
	worldPosition= vec3(worldMatrix *vec4(vertexPosition,1.0));
	///worldNormal = vec3(worldMatrix *vec4(vertexNormal,1.0));
	worldNormal = mat3(transpose(inverse(modelMatrix)))*vertexNormal;
}
