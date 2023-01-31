/*
 * RgtLayerProcessUtil.h
 *
 *  Created on: Jun 8, 2020
 *      Author: l0222891
 */

#ifndef SRC_PROCESS_RGTLAYERPROCESSUTIL_H_
#define SRC_PROCESS_RGTLAYERPROCESSUTIL_H_

#include <vector>

enum SLOPE {
	POSITIF = 1,
	NEGATIF = -1,
	RAIDE = -99
};

class RgtSeed {
public:
	int x, y, z; // position
	int seismicValue; // value used by snap
	int rgtValue; // for memory
	bool operator==(const RgtSeed& other) const {
		return x==other.x && y==other.y && z==other.z && rgtValue==other.rgtValue && seismicValue==other.seismicValue;
	}
};

typedef struct ReferenceDuo {
    std::vector<float> iso;
    std::vector<float> rgt;
} ReferenceDuo;


template<typename InputType>
int	bl_indpol(int ir,InputType *yy,int dimx,int type,int dimx2);

template<typename InputType>
int bl_pointpol(InputType *a,int  ij,int dimx,int imarge);

template<typename InputType>
void UtFiltreMeanX(InputType *tab1,InputType *tab2,std::size_t nx,std::size_t fx,std::size_t opt=1);

template<typename InputType>
void UtFiltreMedianeX(InputType *yy, int idimx, int idimy, int lwx);

template<typename InputType>
void UtFiltreMedianeOrder(InputType*, int);

template<typename InputType>
bool vectorCompare(std::vector<InputType> a, std::vector<InputType> b);

double getNewRgtValueFromReference(long y, long z, long traceIndex, int rgtOriVal, float tdeb, float pasech, long dimy,
                const std::vector<ReferenceDuo>& referenceLayersVec, const std::vector<int>& refValues);

#include "RgtLayerProcessUtil.hpp"

#endif /* SRC_PROCESS_RGTLAYERPROCESSUTIL_H_ */
