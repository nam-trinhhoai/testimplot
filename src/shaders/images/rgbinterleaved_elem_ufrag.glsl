#version 330 core

// vertices datas
in vec2 v_textureCoordinates;

// uniforms
uniform usampler2D rgb_Texture;
uniform usampler2D mask_Texture;

uniform float f_opacity = 0.25; 

uniform float red_rangeMin = 0.0; 
uniform float red_rangeRatio = 0.1;

uniform float green_rangeMin = 0.0; 
uniform float green_rangeRatio = 0.1;

uniform float blue_rangeMin = 0.0; 
uniform float blue_rangeRatio = 0.1;

uniform bool minValueActivated = false;
uniform float minValue = 0.0;

out vec4 f_fragColor; // shader output color

float clip(float val)
{
 	if(val>1.0)
		return 1.0;
	if(val<0.0)
		return 0.0;
	return val;
}

float texture_bilinear_red(in usampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize, float rangeX, float rangeY)
{
    float tl = clip((texture(t, uv).r - rangeX) * rangeY);
    float tr = clip((texture(t, uv + vec2(texelSize.x, 0.0)).r - rangeX) * rangeY);
    float bl = clip((texture(t, uv + vec2(0.0, texelSize.y)).r - rangeX) * rangeY);
    float br = clip((texture(t, uv + vec2(texelSize.x, texelSize.y)).r - rangeX) * rangeY);
      
    
    vec2 f = fract( uv * textureSize );
    float tA = mix( tl, tr, f.x );
    float tB = mix( bl, br, f.x );
    return mix( tA, tB, f.y );
}

float texture_bilinear_green(in usampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize, float rangeX, float rangeY)
{
    float tl = clip((texture(t, uv).g - rangeX) * rangeY);
    float tr = clip((texture(t, uv + vec2(texelSize.x, 0.0)).g - rangeX) * rangeY);
    float bl = clip((texture(t, uv + vec2(0.0, texelSize.y)).g - rangeX) * rangeY);
    float br = clip((texture(t, uv + vec2(texelSize.x, texelSize.y)).g - rangeX) * rangeY);


    vec2 f = fract( uv * textureSize );
    float tA = mix( tl, tr, f.x );
    float tB = mix( bl, br, f.x );
    return mix( tA, tB, f.y );
}

float texture_bilinear_blue(in usampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize, float rangeX, float rangeY)
{
    float tl = clip((texture(t, uv).b - rangeX) * rangeY);
    float tr = clip((texture(t, uv + vec2(texelSize.x, 0.0)).b - rangeX) * rangeY);
    float bl = clip((texture(t, uv + vec2(0.0, texelSize.y)).b - rangeX) * rangeY);
    float br = clip((texture(t, uv + vec2(texelSize.x, texelSize.y)).b - rangeX) * rangeY);


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
    //float redVal = clip((texture(rgb_Texture.r, v_textureCoordinates.st).r - red_rangeMin) * red_rangeRatio);
    //float greenVal = clip((texture(rgb_Texture.g, v_textureCoordinates.st).r - green_rangeMin) * green_rangeRatio);
    //float blueVal = clip((texture(rgb_Texture.b, v_textureCoordinates.st).r - blue_rangeMin) * blue_rangeRatio);
    
    ivec2 texSize = textureSize(rgb_Texture, 0);
    vec2 texelSize = vec2(1.0/texSize);
    float redVal = texture_bilinear_red(rgb_Texture, v_textureCoordinates.st, texSize, texelSize, red_rangeMin, red_rangeRatio);
    
    texSize = textureSize(rgb_Texture, 0);
    texelSize = vec2(1.0/texSize);    
    float greenVal = texture_bilinear_green(rgb_Texture, v_textureCoordinates.st, texSize, texelSize, green_rangeMin, green_rangeRatio);
    
    texSize = textureSize(rgb_Texture, 0);
    texelSize = vec2(1.0/texSize);    
    float blueVal = texture_bilinear_blue(rgb_Texture, v_textureCoordinates.st, texSize, texelSize, blue_rangeMin, blue_rangeRatio);

    if (minValueActivated) {
        vec3 hsv = rgb2hsb(vec3(redVal,greenVal,blueVal));
        if (hsv.z<minValue) {
            hsv.z = minValue;
        }
        vec3 newRgb = hsb2rgb(hsv);
        redVal = newRgb.r;
        greenVal = newRgb.g;
        blueVal = newRgb.b;
    }

    float  mask = texture(mask_Texture, v_textureCoordinates.st).r/255.0;
    f_fragColor = vec4(redVal,greenVal,blueVal,mask *f_opacity);
}
