#include "SeismicPropagator.h"

long SeedsGenericPropagator::DNT_MAX = 1000; /* augmentation en cas de realloc */

long SeedsGenericPropagator::TAS_MAX = 1000; /* augmentation en cas de realloc */

SeedsGenericPropagator::SeedsGenericPropagator() : _tabSeedType(), _tabIso(), _tabAmp() {

}

SeedsGenericPropagator::~SeedsGenericPropagator() {}

long SeedsGenericPropagator::mapWidth() {
	return dimy;
}

long SeedsGenericPropagator::mapHeight() {
	return dimz;
}

/*=============================================================================

Fonction:       UtEnltas

But     :       enlevement du tas de l'element en tete du tas
                avec reamenagement de ce tas

Entree  :

Sortie  :
		vl  (critere de tri)
		in  (indice de reconnaissance de l'element)

Retour  :       nt (nb elements du tas) -> valeur de la fonction

=============================================================================*/

long SeedsGenericPropagator::UtEnltas(long *in, long *vl, long* parent, long* parent_trace_index)
{
    long vtemp,vtemp1,vtemp2,valin;
    long i,j,iok;
    struct tas_i *ptnt,*pti,*ptj;

    i=1;
    pti = ad_tas_i+1;
    *in = (pti)->ind;
    *vl = (pti)->val;
    *parent = (pti)->ind_parent;
    *parent_trace_index = (pti)->parent_trace_index;
    iok=0;
    ptnt = ad_tas_i + nt;
    valin = (ptnt)->val;
    while (iok ==0)
    {
        j=i+i;
        ptj = (ad_tas_i+j);
        if (j < nt)
        {
            vtemp= (ptj)->val;
            vtemp1= (ptj+1)->val;
            if (vtemp1 < vtemp)
            {
                vtemp=vtemp1;
                ++j;
                ++ptj ;
            }
            if (vtemp < valin)
            {
                (pti)->ind = (ptj)->ind;
                (pti)->val = vtemp;
                (pti)->ind_parent = (ptj)->ind_parent;
                (pti)->parent_trace_index = (ptj)->parent_trace_index;
                i=j;
                pti = ptj;
            }
            else
                iok=1;
        }
        else
            iok=1;
    }
    (pti)->ind = (ptnt)->ind;
    (pti)->val = (ptnt)->val;
    (pti)->ind_parent = (ptnt)->ind_parent;
    (pti)->parent_trace_index = (ptnt)->parent_trace_index;
    --nt;

    return nt;

}

/*=============================================================================

Fonction:       UtInstas

But     :       insertion dans le tas d'un element avec un certain
		critere de tri

Entree  :
		in  (indice de reconnaissance de l'element)
		vl  (critere de tri)

Sortie  :

Retour  :       nt (nb elements du tas) -> valeur de la fonction

=============================================================================*/


long SeedsGenericPropagator::UtInstas(long *in,long *vl, long* parent, long* parent_trace_index) {
    long noct;
    struct tas_i *ptnti,*ptnti2;
    /*
       char *malloc(),*realloc();
    */
    long nti,nti2,iok;

    ++nt ;
    if (nt > nt_max )
    {
        if (nt_max == 0)
        {
            nt_max  = DNT_MAX;
            noct = sizeof(struct tas_i)*(nt_max+1);
            ad_tas_i = (struct tas_i *) (malloc(noct));
            if (ad_tas_i == 0) {
                printf("pb malloc tas\n");
                exit(-1);
            }
        }
        else
        {
            nt_max += DNT_MAX;
            noct = sizeof(struct tas_i)*(nt_max+1);
            ad_tas_i = (struct tas_i *) (realloc(ad_tas_i,noct));
            if (ad_tas_i == 0) {
                printf("pb realloc tas\n");
                exit(-1);
            }
        }
    }

    iok = 0;
    nti= nt;
    ptnti = (ad_tas_i+nti);
    while (iok == 0)
    {
        nti2=nti/2;
        ptnti2 = (ad_tas_i+nti2);
        if (nti2 != 0)
        {
            if ((ptnti2)->val > (*vl))
            {
                (ptnti)->ind = (ptnti2)->ind;
                (ptnti)->val = (ptnti2)->val;
                (ptnti)->ind_parent = (ptnti2)->ind_parent;
                (ptnti)->parent_trace_index = (ptnti2)->parent_trace_index;
                nti=nti2;
                ptnti = ptnti2;
            }
            else
                iok=1;
        }
        else
            iok=1;
    }
    (ptnti)->ind = (*in);
    (ptnti)->val = (*vl);
    (ptnti)->ind_parent = (*parent);
    (ptnti)->parent_trace_index = (*parent_trace_index);

    return nt;
}

/*=============================================================================

Fonction:       UtRaztas

But     :       declaration de mise a zero du tas

Entree  :

Sortie  :

Retour  :       nt (nb elements du tas) a 0

=============================================================================*/
long SeedsGenericPropagator::UtRaztas()

{
    nt = 0;
    return nt;
}

/*=============================================================================

Fonction:       UtFreetas

But     :       liberation du tableau ad_tas_i

Entree  :

Sortie  :

Retour  :

=============================================================================*/

void SeedsGenericPropagator::UtFreetas()

{
    delete ad_tas_i;
    ad_tas_i = nullptr;
    nt_max = 0;
    nt = 0;
}

std::vector<int>& SeedsGenericPropagator::getTabSeedType() {
	return _tabSeedType;
}

void SeedsGenericPropagator::setTabSeedType(const std::vector<int>& tab) {
	_tabSeedType = tab;
}

std::vector<float>& SeedsGenericPropagator::getIsochroneTab() {
	return _tabIso;
}

void SeedsGenericPropagator::setIsochroneTab(const std::vector<float>& tab) {
	_tabIso = tab;
}

std::vector<float>& SeedsGenericPropagator::getAmplitudeTab() {
	return _tabAmp;
}

void SeedsGenericPropagator::setAmplitudeTab(const std::vector<float>& tab) {
	_tabAmp = tab;
}

void SeedsGenericPropagator::clearTabs() {
	_tabSeedType.clear();
	_tabIso.clear();
	_tabAmp.clear();
}

std::vector<RgtSeed> SeedsGenericPropagator::staticExtractSeedsFromTabs(long dimx,
		long dimy, long dimz, double sampleRate, double firstSample,
		float* tabIso, float* tabAmp, int seedMaxOutNumber) {
	int IsEdge;
	int nbseeds = 0 ;
	std::vector<RgtSeed> newSeeds;
	long iz, iy, ix, vz, vy;
	float nonVal=-9999;
	std::vector<std::size_t> seenZ;
	std::map<std::size_t, std::size_t> zToSeedMap;
	for(iz = 1; iz < dimz-1; iz +=1) {
		for(iy=1; iy < dimy-1; iy +=1) {
			if(tabIso[iz*dimy+iy] != nonVal) {
				IsEdge = 0 ;
				for (vz=-1; vz <= 1 ; vz ++) {
					for(vy=-1; vy <=1; vy++) {
						if(tabIso[(iz+vz)*dimy + iy+vy] == nonVal) {
							IsEdge = 1 ;

							nbseeds ++ ;
							RgtSeed newSeed;
							ix = (tabIso[iz*dimy + iy] -firstSample)/sampleRate ;
							if (ix<0) {
								ix = 0;
							} else if (ix>=dimx) {
								ix = dimx-1;
							}
							newSeed.x = ix;
							newSeed.y = iy;
							newSeed.z = iz;
							newSeed.seismicValue = tabAmp[iz*dimy + iy];
							newSeed.rgtValue = nonVal;

							//printf(" seed %d ix %d iy %d iz %d \n",nbseeds, ix , iy,iz)  ;
							newSeeds.push_back(newSeed);
							if (seenZ.size()==0 || seenZ[seenZ.size()-1]!=newSeed.z) {
								seenZ.push_back(newSeed.z);
								zToSeedMap[newSeed.z] = newSeeds.size()-1;
							}
						}
						if(IsEdge == 1) break ;
					}
					if(IsEdge == 1) break ;
				}
			}
		}
	}
	int Nfiltered = seedMaxOutNumber;
	if (newSeeds.size()<Nfiltered) {
		return newSeeds;
	} else {
		//int maxSeedBySlice = 20;
		std::vector<RgtSeed> filteredSeeds;
		filteredSeeds.resize(Nfiltered);
		unsigned timeSeed = std::chrono::system_clock::now().time_since_epoch().count();

		std::shuffle (newSeeds.begin(), newSeeds.end(), std::default_random_engine(timeSeed));
		for (std::size_t i=0; i<Nfiltered; i++) {
			filteredSeeds[i] = newSeeds[i];
		}

		return filteredSeeds;
	}
}
