#version 150 core

uniform isampler2D redMap;
uniform isampler2D greenMap;
uniform isampler2D blueMap;

uniform float opacity = 0.25; // tile opacity

uniform vec2 redRange;
uniform vec2 greenRange;
uniform vec2 blueRange;

in vec3 worldPos;
//in vec3 viewPos;
//in vec3 position;
//in vec3 normal;
in vec3 worldNormal;
in vec2 texCoord;
//in vec3 lightA;
//in vec3 lightB;
in float potentialHole;
in float height;



//vec3 lightPosition = vec3(-800,2000,-3500);
//vec3 lightIntensity = vec3(1.0,1.0,1.0);

out vec4 fragColor;
/*
float clip(float val)
{
 	if(val>1.0)
		return 1.0;
	if(val<0.0)
		return 0.0;
	return val;
}*/

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


void main()
{	
	
	if(potentialHole > 0.0) {
		discard;
	}
	//float redval = clip((texture(redMap, texCoord.st).r - redRange.x) *redRange.y);
	//float greenval = clip((texture(greenMap, texCoord.st).r - greenRange.x) *greenRange.y);
	//float blueval = clip((texture(blueMap, texCoord.st).r - blueRange.x) *blueRange.y);

	//fragColor = vec4(redval,greenval,blueval,opacity);

	ivec2 texSize = textureSize(redMap, 0);
	vec2 texelSize = vec2(1.0/texSize);
	float redval = texture_bilinear(redMap, texCoord.st, texSize, texelSize, redRange.x, redRange.y);

	texSize = textureSize(greenMap, 0);
	texelSize = vec2(1.0/texSize);    
	float greenval = texture_bilinear(greenMap, texCoord.st, texSize, texelSize, greenRange.x, greenRange.y);

	texSize = textureSize(blueMap, 0);
	texelSize = vec2(1.0/texSize);    
	float blueval = texture_bilinear(blueMap, texCoord.st, texSize, texelSize, blueRange.x, blueRange.y);


	// SIMPLE PHONG SHADING
	


	// ambient
	vec3 ambient = vec3(0.5, 0.5, 0.5);


	// diffuse
	//vec3 normalN = normalize(normal);
	vec3 worldNormalN = normalize(worldNormal);
	vec3 lightPosA = vec3(0,-8000,0);
	//vec3 lightPosB = vec3(-3000,-5000,0);

	

	//vec3 lightDirA = vec3(-0.8,-1.0,0.0);
	vec3 lightDirA = normalize(lightPosA - worldPos);
	//vec3 lightDirB = normalize(lightPosB - worldPos);

	vec3 diffuseA = 0.5 * vec3(1,1,1) * max(dot(worldNormalN,normalize(lightDirA)), 0.0);
//	vec3 diffuseB = 0.0 * vec3(0,0,1) * max(dot(worldNormalN,lightDirB), 0.0);

	//vec3 diffuse = clamp(diffuseA + diffuseB, 0.0, 1.0);
	
	// specular	
/*	vec3 viewDir = normalize(-viewPos) ;
	float shininess = 64;   

	vec3 reflectDirA = normalize(reflect(-lightDirA, normalN));
	vec3 reflectDirB = normalize(reflect(-lightDirB, normalN));
	vec3 specA = vec3(1,1,1) * pow(max(dot(viewDir, reflectDirA), 0.0), shininess);
	vec3 specB = vec3(1,1,1) * pow(max(dot(viewDir, reflectDirB), 0.0), shininess);


	vec3 spec = clamp(specA + specB, 0.0, 1.0);	*/
	
	vec3 objectColor = vec3(redval,greenval,blueval);	
	vec3 result = ambient * objectColor + diffuseA * objectColor;// + spec;
	
//vec3 result = adsModel(viewPos,normalN, objectColor);


   	fragColor = vec4(result, opacity);//result
   	//fragColor = vec4(height/1000.0, 0, 0, opacity);

	gl_FragDepth = gl_FragCoord.z;
}

