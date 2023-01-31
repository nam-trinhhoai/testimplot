#version 150 core

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexTexCoord;

out vec3 worldPos;
out vec3 viewPos;
out vec3 position;
out vec3 normal;
out vec2 texCoord;
out vec3 lightA;
out vec3 lightB;
out float potentialHole;
out float height;

uniform mat4 modelMatrix;
uniform mat4 modelView;
uniform mat3 modelViewNormal;
uniform mat4 mvp;
uniform mat4 viewMatrix;

uniform sampler2D surfaceMap;
uniform float cubeOrigin;
uniform float cubeScale;
uniform float heightThreshold;

float rand(float n){return fract(sin(n) * 43758.5453123);}


float getHeight(vec2 texC)
{
   return texture(surfaceMap, texC.st).r;
}


void main()
{
	texCoord = vertexTexCoord;
	float origval = texture(surfaceMap, texCoord.st).r;
	height = origval;
	if(origval < heightThreshold) {
		potentialHole = 0.0;
	} else {
		potentialHole = 1.0;
	}

	ivec2 texSize = textureSize(surfaceMap, 0);
	vec2 texelSize = vec2(1.0 / texSize);

	// compute normal from heightMap
	float x = getHeight(vec2(texCoord.s - texelSize.x, texCoord.t)) - getHeight(vec2(texCoord.s + texelSize.x, texCoord.t));
	float y = 2.0;
	float z = getHeight(vec2(texCoord.s, texCoord.t + texelSize.y)) - getHeight(vec2(texCoord.s, texCoord.t - texelSize.y));


	vec3 normalH = normalize(vec3(x, y, z) / 2.0);

	vec3 newPos = vertexPosition;
	newPos.y = cubeOrigin + cubeScale * origval;


	// transform to view space   

	normal = normalize(modelViewNormal * normalH);
	//normal = normalH;

	vec3 lightPosA = vec3(2500,200,500);
	vec3 lightPosB = vec3(-800,2000,-3500);

	lightA = (viewMatrix * vec4(lightPosA, 1.0)).xyz;
	lightB = (viewMatrix * vec4(lightPosB, 1.0)).xyz;

	worldPos = vec3(modelMatrix * vec4(newPos, 1.0));
	position = vec3(modelView * vec4(newPos, 1.0));
	viewPos = vec3(modelView * vec4(newPos, 1.0));
	gl_Position = mvp * vec4(newPos, 1.0);
}
