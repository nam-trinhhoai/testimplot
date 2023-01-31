

#include <stdio.h>
#include <malloc.h>

#include <util.h>
#include <memoryManager.h>
#include <horizonUtils.h>

bool horizonRead(std::string filename, int dimx, int dimy, int dimz,
		float pasech, float tdeb, float **horizon)
{
	int err = 0;
	long size0 = (long)dimy*dimz;
	FILE *pFile = nullptr;
	pFile = fopen((char*)filename.c_str(), "r");
	if ( pFile == nullptr )
	{
		fprintf(stderr, "problem to read horizon file %s\n", (char*)filename.c_str());
		return false;
	}
	int ret = mallocSafe((void**)&(*horizon), size0, sizeof(float));
	if ( ret != SUCCESS )
	{
		fprintf(stderr, "error in allocation memory in %s %d - size: %d\n", __FILE__, __LINE__, size0);
		return false;
	}
	fread(*horizon, sizeof(float), size0, pFile);
	fclose(pFile);
	for (long add=0; add<size0; add++)
		(*horizon)[add] = ((*horizon)[add]-tdeb)/pasech;
	return true;
}


float horizonMean(float *horizon, int dimy, int dimz)
{
	if ( horizon == nullptr ) return 0.0f;
	double mean = 0.0;
	for (long add=0; add<(long)dimy*dimz; add++)
		mean += horizon[add];
	return (float)(mean/(double)((double)dimy*dimz));
}
