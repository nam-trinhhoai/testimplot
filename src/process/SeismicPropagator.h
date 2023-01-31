#ifndef MURATPROCESSLIB_SRC_SEISMICPROPAGATOR_H_
#define MURATPROCESSLIB_SRC_SEISMICPROPAGATOR_H_

#include "RgtLayerProcessUtil.h"
//#include "Cube.h"

#include <cmath>
#include <cstdio>
#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/errno.h>
#include <tuple>
#include <memory>
#include <vector>

class Seismic3DDataset;

class SeedsGenericPropagator {
public:
	static long DNT_MAX; /* augmentation en cas de realloc */
	static long TAS_MAX; /* augmentation en cas de realloc */

	struct tas_i {
		long val;
		long ind;
		long ind_parent;
		long parent_trace_index;
	};

	enum Retour {
		Ok = 0,
		Error = 1
	};
	SeedsGenericPropagator();
	virtual ~SeedsGenericPropagator();

	std::vector<int>& getTabSeedType();
	void setTabSeedType(const std::vector<int>&);
	std::vector<float>& getIsochroneTab();
	void setIsochroneTab(const std::vector<float>&);
	std::vector<float>& getAmplitudeTab();
	void setAmplitudeTab(const std::vector<float>&);
	void clearTabs();

	long mapWidth();
	long mapHeight();

	virtual std::vector<RgtSeed> extractSeedsFromTabs(int seedMaxOutNumber=1000) = 0;
	static std::vector<RgtSeed> staticExtractSeedsFromTabs(long dimX, long dimY, long dimZ,
			double sampleRate, double firstSample, float* tabIso, float* tabAmp,
			int seedMaxOutNumber);

protected:
	//long UtEnltasin(long *in) ;
	long UtEnltas(long *in, long *vl, long* parent, long* parent_trace_index) ;
	long UtInstas(long *in,long *vl, long* parent, long * parent_trace_index) ;
	long UtRaztas() ;
	void UtFreetas() ;

	//static int compPt(const void *v1,const void *v2) ;

	//static Retour PgSpline (float*,float*,int, float,float, float*);
	//static Retour PgSplineValue(float*,float*,float*,int,float,float*);

protected:
	struct tas_i *ad_tas_i=nullptr;
	long nt_max=0,nt=0;

	std::vector<int> _tabSeedType;
	std::vector<float> _tabIso;
	std::vector<float> _tabAmp;
	long dimx, dimy, dimz;
	float tdeb=0, pasech=1;
};

template<typename InputType>
class SeedsPropagator : public SeedsGenericPropagator {
public:
	SeedsPropagator(Seismic3DDataset* seismic, int channel);
	virtual ~SeedsPropagator();
	// tabSeedType MUST be set with 1 for old seeds and 2 for new seeds
	std::vector<RgtSeed> propagate(
			const std::vector<RgtSeed>& originSeeds, int seedTypePropagate=2, int type_D=1,
			int sizeCorr=10, int isx=5, float seuilCorr=0.8, int numIter = 3, int seedMaxOutNumber=1000);
	virtual std::vector<RgtSeed> extractSeedsFromTabs(int seedMaxOutNumber=1000) override;

private:
	static Retour SortTab(InputType *a, int lw) ;
	static int bl_indpol(long *ir,InputType *yy,long  *dimx,long  *dimy,long *dimz,int *type,int *dimx2);
	static int bl_pointpol(InputType *a, int *ij, long *dimx, int *imarge);

	static float UtCorrTr(InputType *yy, InputType *yyprec,int ir,int irp,int ideb,int ifin) ;

	static void copyContentWithoutDimV(const InputType* ori, InputType* out, long outDim, int dimV, int channel);

	Seismic3DDataset* cubeAmp;
	int channel;
};

#include "SeismicPropagator.hpp"

#endif // MURATPROCESSLIB_SRC_SEISMICPROPAGATOR_H_
