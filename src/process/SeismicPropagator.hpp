#ifndef MURATPROCESSLIB_SRC_SEISMICPROPAGATOR_HPP_
#define MURATPROCESSLIB_SRC_SEISMICPROPAGATOR_HPP_

#include "SeismicPropagator.h"
#include "seismic3ddataset.h"
#include "affinetransformation.h"

#include <QDebug>

#include <algorithm>
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

template<typename InputType>
SeedsPropagator<InputType>::SeedsPropagator(Seismic3DDataset* seismic, int _channel) :
SeedsGenericPropagator() {

	cubeAmp = seismic;
	channel = _channel;

    dimx = cubeAmp->height();
    dimy = cubeAmp->width();
    dimz = cubeAmp->depth();
    tdeb = cubeAmp->sampleTransformation()->b();
    pasech = cubeAmp->sampleTransformation()->a();

}

template<typename InputType>
SeedsPropagator<InputType>::~SeedsPropagator() {

}

template<typename InputType>
void SeedsPropagator<InputType>::copyContentWithoutDimV(const InputType* ori,
		InputType* out, long outDim, int dimV, int channel) {
	for (long idx=0; idx<outDim; idx++) {
		out[idx] = ori[idx*dimV+channel];
	}
}

template<typename InputType>
std::vector<RgtSeed> SeedsPropagator<InputType>::propagate(const std::vector<RgtSeed>& originSeeds,
										int seedTypePropagate, int type_D, int sizeCorr, int isx,
										float seuilCorr, int Nbiter, int seedMaxOutNumber) {
    int nbThread=8,useMask;
    int i,j,k,taille=1;


    printf(" type %d fen %d sx %d seuilCorre %f\n",type_D, sizeCorr, isx, seuilCorr) ;
    InputType *image_don=NULL,*temp=NULL;

    long ind;
    int ii,jj,l,CodeRetour;
    long ir;
    long irj;
    int ln  ;
    long ix,iy,iz,num;
    char *image_label = NULL ;
    float *tabIso = NULL ;
    float *tabAmp = NULL ;
    InputType *tab2=NULL ;
    float nonVal = -9999;
    //int seedTypePropagate = 2; //2  only new seeds, 1 all seeds wil be propagate

    std::vector<InputType> _image_don;
    _image_don.resize(2*isx+2*sizeCorr+1, 0);
    image_don = _image_don.data() ;

    std::vector<std::vector<InputType>> seedsAmp;

    _tabIso.resize(dimy*dimz, nonVal);
    tabIso = _tabIso.data();
    _tabAmp.resize(dimy*dimz, nonVal);
    tabAmp = _tabAmp.data();
    _tabSeedType.resize(dimy*dimz, 0);
    int* tabSeedType = _tabSeedType.data();

    long NBtas = 0;
    long longue = 0;
    long ijk0,ind_seed ;
    long nt_max=0,nt=0,indc;
    long critere,valCorr ;

	std::vector<InputType> seedArray;
	seedArray.resize(dimx);
	std::vector<InputType> _seedArray;
	_seedArray.resize(dimx*cubeAmp->dimV());
    for (const RgtSeed& seed : originSeeds) {
    	cubeAmp->readSubTraceAndSwap(_seedArray.data(), 0, dimx, seed.y, seed.z);
    	copyContentWithoutDimV(_seedArray.data(), seedArray.data(), dimx, cubeAmp->dimV(), channel);

		//printf(" Seed ix %d iy %d iz %d\n", seed.x,seed.y,seed.z) ;
		ijk0 = /*static_cast<long>(seed.z)*dimx*dimy + seed.y*dimx +*/ seed.x ;
		long dim1 = 1;
		ind_seed = SeedsPropagator<InputType>::bl_indpol(&ijk0,seedArray.data(), &dimx,&dim1,&dim1,&type_D, &isx) ;
		//printf("image_dom[%d] %d image_dom[%d] %d\n", ijk0, seedArray[ijk0], ind_seed, seedArray[ind_seed] );

		valCorr = 0 ;
		//NBtas=UtInstas( &ind_seed, &valCorr) ;
		tabIso[static_cast<long>(seed.z)*dimy + seed.y] = tdeb + (ind_seed%dimx)*pasech  ;
		tabAmp[static_cast<long>(seed.z)*dimy + seed.y] = seedArray[ind_seed] ;
		if (tabSeedType[static_cast<long>(seed.z)*dimy + seed.y]!=1) {
			tabSeedType[static_cast<long>(seed.z)*dimy + seed.y] = seedTypePropagate;
		}
    }

	long posx,posy,posz,ijk, ijk_parent,parent_trace_index;
    int iter,vy,vz,IsEdge ;
    std::vector<InputType> tmpBuffer;
    tmpBuffer.resize((2*isx+2*sizeCorr+1)*cubeAmp->dimV());
    for(iter =0; iter < Nbiter ; iter ++) {
        seedsAmp.clear();
        long seedsAmpIndex = 0;
        // une iteration de plus a partir des bords de la premiere propoagation
        for(iz = 1; iz < dimz-1; iz ++) {
            for(iy=1; iy < dimy-1; iy ++) {
                if(tabIso[iz*dimy+iy] != nonVal && tabSeedType[iz*dimy + iy]==seedTypePropagate) {
                    IsEdge = 0 ;
                    for (vz=-1; vz <= 1 ; vz ++) {
                        for(vy=-1; vy <=1; vy++) {
                            if(tabIso[(iz+vz)*dimy + iy+vy] == nonVal) {
                                IsEdge = 1 ;
                                seedsAmp.push_back(std::vector<InputType>());
								seedsAmp.back().resize(2*sizeCorr+1);
								if (iter!=0) {
									int xmin = ((tabIso[iz*dimy+iy]-tdeb)/pasech)-sizeCorr;
									int init_xmin = xmin;
									int xmax_1 = xmin + 2*sizeCorr + 1;
									if (xmin<0) {
										xmin=0;
									} else if (xmin>=dimx) {
										xmin = dimx - 1;
									}
									if (xmax_1<0) {
										xmax_1 = 0;
									} else if (xmax_1>dimx) {
										xmax_1 = dimx;
									}
									tmpBuffer.resize((2*sizeCorr+1)*cubeAmp->dimV(), 0);
									cubeAmp->readSubTraceAndSwap(tmpBuffer.data() + (xmin-init_xmin), xmin, xmax_1, iy, iz);
									copyContentWithoutDimV(tmpBuffer.data(), seedsAmp.back().data(), 2*sizeCorr+1, cubeAmp->dimV(), channel);
									ind_seed = iz*dimx*dimy + iy*dimx + static_cast<long>((tabIso[iz*dimy+iy]-tdeb)/pasech) ;
								} else {
									int xmin = ((tabIso[iz*dimy+iy]-tdeb)/pasech)-sizeCorr-isx;
									int init_xmin = xmin;
									int xmax_1 = xmin + 2*sizeCorr + 2*isx+ 1;
									if (xmin<0) {
										xmin=0;
									} else if (xmin>=dimx) {
										xmin = dimx - 1;
									}
									if (xmax_1<0) {
										xmax_1 = 0;
									} else if (xmax_1>dimx) {
										xmax_1 = dimx;
									}
									std::vector<InputType> tmpTab;
									tmpTab.resize(2*isx+2*sizeCorr+1, 0);
									tmpBuffer.resize((2*isx+2*sizeCorr+1)*cubeAmp->dimV(), 0);
									cubeAmp->readSubTraceAndSwap(tmpBuffer.data() + (xmin-init_xmin), xmin, xmax_1, iy, iz);
									copyContentWithoutDimV(tmpBuffer.data(), tmpTab.data(), 2*isx+2*sizeCorr+1, cubeAmp->dimV(), channel);
									long oldInd = isx+sizeCorr;
									long dim1 = 1;
									long reducedDim = 2*isx+2*sizeCorr+1;
									long newInd = bl_indpol(&oldInd, tmpTab.data(), &reducedDim, &dim1, &dim1, &type_D, &isx);
									if (newInd==RAIDE || abs(newInd-oldInd)>=isx) {
										qDebug() << "Unexpected diff";
									}
									memcpy(seedsAmp.back().data(), tmpTab.data()+ isx + (newInd-oldInd), sizeof(InputType)*(2*sizeCorr+1));
									ind_seed = iz*dimx*dimy + iy*dimx + static_cast<long>((tabIso[iz*dimy+iy]-tdeb)/pasech) + (newInd-oldInd) ;
								}
                                //ind_seed = iz*dimx*dimy + iy*dimx + (tabIso[iz*dimy+iy]-tdeb)/pasech ;
                                valCorr = 1000 ;
                                NBtas=UtInstas( &ind_seed, &valCorr, &ind_seed, &seedsAmpIndex) ;

                                //qDebug() << "seeds init " << (ind_seed%dimx);
                                seedsAmpIndex++;
                                longue ++ ;
                            }
                            if(IsEdge == 1) break ;
                        }
                        if(IsEdge == 1) break ;
                    }
                }
            }
        }
        //qDebug() << "Propagator in loop size" << NBtas;
        while(NBtas != 0 )
        {
        	NBtas=UtEnltas(&ijk, &critere, &ijk_parent, &parent_trace_index) ;

            posz = ijk/(dimx*dimy) ;
            posx = (ijk - posz*dimx*dimy)  % dimx ;
            posy = (ijk-posz*dimx*dimy)/dimx  ;
            //qDebug() << "-- posx" << posx << ", posy" << posy << ", posz" << posz << ", ijk" << ijk;
            for (iz=posz-taille; iz<=posz+taille; iz++) {
            	if (iz>=0 && iz<dimz) {
					for (iy=posy-taille; iy<=posy+taille; iy++) {
						if (iy>=0 && iy<dimy) {
							if(tabIso[iz*dimy+iy] == nonVal) {
								memset(image_don, 0, sizeof(InputType)*(2*isx+2*sizeCorr+1));
								int xmin = posx-isx-sizeCorr;
								int init_xmin = xmin;
								int xmax_1 = xmin + 2*isx+2*sizeCorr+1;
								if (xmin<0) {
									xmin=0;
								} else if (xmin>=dimx) {
									xmin = dimx - 1;
								}
								if (xmax_1<0) {
									xmax_1 = 0;
								} else if (xmax_1>dimx) {
									xmax_1 = dimx;
								}
								tmpBuffer.resize((2*isx+2*sizeCorr+1)*cubeAmp->dimV(), 0);
								cubeAmp->readSubTraceAndSwap(tmpBuffer.data() + (xmin-init_xmin), xmin, xmax_1, iy, iz);
								copyContentWithoutDimV(tmpBuffer.data(), image_don, 2*isx+2*sizeCorr+1, cubeAmp->dimV(), channel);
								ir = isx+sizeCorr;
								long dim1 = 1;
								long dimReduced = 2*isx+2*sizeCorr+1;
								irj = bl_indpol(&ir, image_don, &dimReduced, &dim1, &dim1, &type_D, &isx) ;

								if (irj != RAIDE && abs(ir - irj) < isx ) {
									valCorr = 1000*UtCorrTr(seedsAmp[parent_trace_index].data(), image_don, sizeCorr,irj, -sizeCorr, sizeCorr) ;
									//qDebug() << "valCorr" << valCorr;
									if(valCorr > 1000*seuilCorr) {
										longue++;
										long old_irj = irj; // DEBUG
										irj = iz*dimx*dimy + iy*dimx + posx-isx-sizeCorr + irj;
										tabIso[iz*dimy+iy] = tdeb + (irj%dimx)*pasech ;
										tabAmp[iz*dimy+iy] = image_don[old_irj] ;
										valCorr = 1000 - valCorr ;

										/*if (irj%dimx<50) {
											qDebug() << "Aie";
										}
										qDebug() << "irj watcher " << (irj%dimx);*/
										NBtas=UtInstas( &irj, &valCorr, &ijk_parent, &parent_trace_index) ;
										tabSeedType[iz*dimy+iy] = seedTypePropagate;
									}
								}
							}
						}
					}
            	}
            }
        }
        printf(" iter %d longue %d \n",iter,longue) ;
    }/*
    int nbseeds = 0 ;
    std::vector<RgtSeed> newSeeds;
    std::vector<bool> tabSeedSelected;
    tabSeedSelected.resize(dimy*dimz, false);
	for(iz = 1; iz < dimz-1; iz += 1) {
		for(iy=1; iy < dimy-1; iy +=1) {
			if(tabIso[iz*dimy+iy] != nonVal) {
				IsEdge = 0 ;
				for (vz=-1; vz <= 1 ; vz ++) {
					for(vy=-1; vy <=1; vy++) {
						if(tabIso[(iz+vz)*dimy + iy+vy] == nonVal) {
							IsEdge = 1 ;
							if (tabSeedSelected[iz*dimy+iy]==false) {
								for (int seed_vz=iz-seedReductionWindow; seed_vz<=iz+seedReductionWindow; seed_vz++) {
									for (int seed_vy=iy-seedReductionWindow; seed_vy<=iy+seedReductionWindow; seed_vy++) {
										if (seed_vz>=0 && seed_vz<dimz &&
											seed_vy>=0 && seed_vy<dimy) {
											tabSeedSelected[seed_vz*dimy+seed_vy]=true;
										}
									}
								}

								nbseeds ++ ;
								RgtSeed newSeed;
								ix = (tabIso[iz*dimy + iy] -tdeb)/pasech ;
								newSeed.x = ix;
								newSeed.y = iy;
								newSeed.z = iz;
								newSeed.seismicValue = tabAmp[iz*dimy + iy];
								newSeed.rgtValue = nonVal;
								//printf(" seed %d ix %d iy %d iz %d \n",nbseeds, ix , iy,iz)  ;
								newSeeds.push_back(newSeed);
							}
							tabSeedType[iz*dimy+iy] = 1;
						}
						if(IsEdge == 1) break ;
					}
					if(IsEdge == 1) break ;
				}
			}
		}
	}*/
    std::vector<RgtSeed> newSeeds = extractSeedsFromTabs(seedMaxOutNumber);
	qDebug() << "Propagator output size" << newSeeds.size();

    UtFreetas() ;
    return (newSeeds);
}

template<typename InputType>
std::vector<RgtSeed> SeedsPropagator<InputType>::extractSeedsFromTabs(int seedMaxOutNumber) {
	int IsEdge;
	int nbseeds = 0 ;
	std::vector<RgtSeed> newSeeds;
	//std::vector<bool> tabSeedSelected;
	//tabSeedSelected.resize(dimy*dimz, false);
	float* tabIso = _tabIso.data();
	float* tabAmp = _tabAmp.data();
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
							//if (tabSeedSelected[iz*dimy+iy]==false) {
								/*for (int seed_vz=iz-seedReductionWindow; seed_vz<=iz+seedReductionWindow; seed_vz++) {
									for (int seed_vy=iy-seedReductionWindow; seed_vy<=iy+seedReductionWindow; seed_vy++) {
										if (seed_vz>=0 && seed_vz<dimz &&
											seed_vy>=0 && seed_vy<dimy) {
											tabSeedSelected[seed_vz*dimy+seed_vy]=true;
										}
									}
								}*/

								nbseeds ++ ;
								RgtSeed newSeed;
								ix = (tabIso[iz*dimy + iy] -tdeb)/pasech ;
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
							//}
							_tabSeedType[iz*dimy+iy] = 1;
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

		/*std::shuffle(seenZ.begin(), seenZ.end(), std::default_random_engine(timeSeed));
		std::size_t z_index = 0;
		std::size_t indexFiltered = 0;
		while (indexFiltered<Nfiltered && z_index<seenZ.size()) {
			std::size_t indexSeed = zToSeedMap[seenZ[z_index]];
			int zSeedMin = newSeeds[indexSeed].z;
			while (newSeeds[indexSeed].z==zSeedMin && indexSeed<newSeeds.size() && indexFiltered<Nfiltered) {
				filteredSeeds[indexFiltered] = newSeeds[indexSeed];
				indexFiltered ++;
				indexSeed++;
			}

			z_index ++;
		}*/



		std::shuffle (newSeeds.begin(), newSeeds.end(), std::default_random_engine(timeSeed));
		for (std::size_t i=0; i<Nfiltered; i++) {
			filteredSeeds[i] = newSeeds[i];
		}

		return filteredSeeds;
	}
}

template<typename InputType>
float SeedsPropagator<InputType>::UtCorrTr(InputType *yy, InputType *yyprec,int ir,int irp,int ideb,int ifin)
{
    double  som = 0, som2 = 0, som3 = 0;
    int i,nbech=0;
    float   coef = 0.0;
    double somX = 0, somY = 0 ;
    for(i = ideb; i <= ifin ; i ++) {
        /*
        som += (yyprec[ ir + i] - yy[irp + i])*(yyprec[ ir + i] - yy[irp + i]);
        */
        nbech++;
        som += yyprec[ irp + i] * yy[ir + i];
        som2 += yyprec[ irp +i] * yyprec[ irp +i];
        som3 += yy[ir + i] * yy[ir + i];
        somX += yyprec[ irp +i] ;
        somY += yy[ir + i];
    }
    if (som2 * som3 != 0.)
    {
        /*
        coef = 1.0 - som / (sqrt(som2) * sqrt(som3));
        */
        som = som - somX*somY/nbech ;
        som2=som2 - somX*somX/nbech ;
        som3=som3 - somY*somY/nbech ;

        //coef = som / (std::sqrt(som2) * std::sqrt(som3));

        coef = 0.7*som / (sqrt(som2 ) * sqrt(som3 )) +  0.3*2*(som  / (som2 + som3 ));

    }
    else
    {
        coef = 0.0;
    }
    return(coef);
}

/*----------------------FONCTION C-------SISMAGE-----ELF AQUITAINE----
C
C                                    APPLICATION:  COMMUN/utilsub
C   NOM DE LA FONCTION :  bl_indpol.c
C
C   OBJECTIF GENERAL : Fonction retournant l'indice sur le sommet
C                      le plus proche dans la bonne polarite
C                      tenant compte des traces raides.
C
C-----------------DESCRIPTION DES PARAMETRES D'APPEL------------------
C
C  INPUT :
C     ir     : Indice entree dans le tableau 2D yy
C                ATTENTION le debut du tableau est a ir=0 (convention C)
C     yy     : Tableau 2D
C     dimx  : Dimension en x de la matrice 2D
C     dimy  : Dimension en y de la matrice 2D
C     type   : Type de pointe sur extrema positif ou negatif
C               =-1  negatif       =+1 positif
C     dimx2 : Dimension en x maxi pour la recherche des plateaux
C
C  OUTPUT :
C           Code retour de la fonction contenant l'indice
C           du sommet le plus proche dans la bonne polarite
C             si trace raide: code retour = -99
C
C---------------------------------------------------------------------*/

template<typename InputType>
int	SeedsPropagator<InputType>::bl_indpol(long *ir,InputType *yy,long  *dimx,long  *dimy,long *dimz,int *type,int *dimx2)
{
    int	ii, ij;
    int	ind_y, ij_min, ij_max;

    ij = *ir;
    ind_y = ij / *dimx;
    ij_min = ind_y * *dimx;
    ij_max = ij_min + *dimx - 1;

    /* Blindage */
    if(*dimx2 > *dimx / 2)
        *dimx2 = *dimx2 / 2;

    /* Test de la valeur non nulle de la colonne courante */
    ii = SeedsPropagator<InputType>::bl_pointpol(yy, &ij, dimx, dimx2);

    /* Rejet trace raide */
    if(ii == RAIDE ) {
        return(RAIDE);
    }
    if(ii == NEGATIF) {
        /* Recherche sur une polarite negative: pente decroissante */
        if(*type == -1 ) {
            /* Pointe sur minimum local */
            while( ij < ij_max &&
                    (ii = SeedsPropagator<InputType>::bl_pointpol(yy, &ij, dimx, dimx2)) == NEGATIF ) {
                ij++;
            }
            return(ij - 1);
        } else {
            /* Pointe sur maximum local */
            while( ij > ij_min &&
                    (ii = SeedsPropagator<InputType>::bl_pointpol(yy, &ij, dimx, dimx2)) == NEGATIF ) {
                ij--;
            }
            return(ij);
        }
    } else {
        /* Recherche sur une polarite positive: pente croissante */
        if(*type == 1) {
            /* Pointe sur maximum local */
            while( ij < ij_max &&
                    (ii = SeedsPropagator<InputType>::bl_pointpol(yy, &ij, dimx, dimx2)) == POSITIF ) {
                ij++;
            }
            return(ij - 1);
        } else {
            /* Pointe sur minimum local */
            while( ij > ij_min &&
                    (ii = SeedsPropagator<InputType>::bl_pointpol(yy, &ij, dimx, dimx2)) == POSITIF ) {
                ij--;
            }
            return(ij);
        }
    }
}
/*----------------------FONCTION C-------SISMAGE-----ELF AQUITAINE----
C
C   NOM DE LA FONCTION :  bl_pointpol
C
C   OBJECTIF GENERAL :detection de la polarite d'un point
C                       retour special pour les traces raides
C
C-----------------DESCRIPTION DES PARAMETRES D'APPEL------------------
C
C  INPUT :
C            a  : tableau entier 16 bits bloc 2d
C            ij : indice du point a traiter dans le tableau
C                ATTENTION le debut du tableau est a ij=0 (convention C)
C            dimx : nombre d'echantillons d'une ligne
C            imarge: marge maximum de recherche de plateau
C
C  OUTPUT :
C            polarite : retour de la fonction
C                  positif = 0      negatif = 1    raide = -99
C
C---------------------------------------------------------------------*/
/*
#define POSITIF 0
#define NEGATIF 0xffffffff
#define RAIDE   (-99)
*/
template<typename InputType>
int SeedsPropagator<InputType>::bl_pointpol(InputType *a, int *ij, long *dimx, int *imarge)
{
    /***************************************************************/
    /*   isign   : signe du produit du point precedent et du point */
    /*                                                    suivant  */
    /*   pmoins  : Difference avec le point precedent              */
    /*   pplus   : Difference avec le point suivant                */
    /*   ideb    : Indice de detection de debut d'un plateau       */
    /*   ifin    : Indice de detection de fin   d'un plateau       */
    /*   idebx   : Indice de detection de debut de la ligne        */
    /*   ifinx   : Indice de detection de fin   de la ligne        */
    /***************************************************************/
    int	isign, pmoins, ideb, ifin, idebx, ifinx;
    int	pplus;

    pmoins = a[(*ij)] - a[(*ij)-1];
    isign = pmoins * (a[(*ij)] - a[(*ij)+1]);
    idebx = (*ij / *dimx) * (*dimx);
    ifinx = idebx + *dimx - 1;

    if (isign < 0 ) {
        /* le point ij est sur une pente */
        if (pmoins > 0)
            return (POSITIF); /* pente positive */
        else
            return (NEGATIF); /* pente negative */
    } else if (isign > 0) {
        /* le point ij est un extrema pur */
        if (pmoins < 0)
            return (NEGATIF); /* extrema positif */
        else
            return (POSITIF); /* extrema negatif */
    } else {
        /* le point ij est sur un plateau */
        /* detection de la fin et du debut du plateau */
        for (ideb = *ij;
                (ideb>idebx+1) && (a[ideb]-a[ideb-1]) == 0 && (*ij-ideb < *imarge);
                ideb--);
        for (ifin = *ij;
                (ifin<ifinx-1) && (a[ifin+1]-a[ifin]) == 0 && (ifin- *ij < *imarge);
                ifin++);
        pmoins = a[(*ij)] - a[ideb-1];
        pplus = a[*ij] - a[ifin+1];
        isign = pplus * pmoins;
        if (pmoins == 0 || pplus == 0) {
            if (pmoins != 0) {
                if ( pmoins > 0 )
                    return (POSITIF); /* le plateau est sur pente positive */
                else
                    return (NEGATIF); /* le plateau est sur pente negative */
            } else if (pplus != 0) {
                if ( pplus > 0 )
                    return (NEGATIF); /* le plateau est sur pente negative */
                else
                    return (POSITIF); /* le plateau est sur pente positive */
            } else
                return (RAIDE); /* le plateau est une trace raide  */
        }
        if (isign < 0) {
            /* le plateau est sur une pente */
            if ( pmoins > 0 )
                return (POSITIF); /* le plateau est sur pente positive */
            else
                return (NEGATIF); /* le plateau est sur pente negative */
        } else {
            /* le plateau est sommet */
            if ( pmoins > 0 ) {
                /* le plateau est un sommet positif */
                if ( (*ij) - ideb < ifin - (*ij) )
                    return (POSITIF);/* ij dans la premiere moitie */
                else
                    return (NEGATIF); /* ij dans la deuxieme moitie */
            } else if (pmoins < 0) {
                /* le plateau est sur sommet negatif */
                if ( (*ij) - ideb < ifin - (*ij) )
                    return (NEGATIF); /* ij dans la premiere moitie */
                else
                    return (POSITIF);/* ij dans la deuxieme moitie */
            }
        }
    }
}

#endif // MURATPROCESSLIB_SRC_SEISMICPROPAGATOR_HPP_
