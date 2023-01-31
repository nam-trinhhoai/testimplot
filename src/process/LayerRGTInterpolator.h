#ifndef TARUMAPP_SRC_PROCESS_LAYERRGTINTERPOLATOR_H_
#define TARUMAPP_SRC_PROCESS_LAYERRGTINTERPOLATOR_H_

#include "RgtLayerProcessUtil.h"
#include "ioutil.h"
#include <vector>

template<typename RgtType, typename SeismicType>
void layerRGTInterpolatorMultiSeed(const std::vector<float>& inputLayer, long dtauReference, std::vector<float>& outputLayerIso,
		std::vector<float>& outputLayerSeismic, std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers,
		std::vector<RgtSeed> seeds, const Seismic3DDataset* rgt, int channelT, const Seismic3DDataset* seismic, int channelS, bool useSnap,
		bool useMedian, int lwx, int distancePower, int snapWindow, int polarity, float tdeb, float pasech);

template<typename RgtType, typename SeismicType>
void layerRGTInterpolatorMultiSeedCwt(const std::vector<float>& inputLayer, long dtauReference, std::vector<float>& outputLayerIso,
		std::vector<float>& outputLayerSeismic,std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers,
		std::vector<RgtSeed> seeds, const Seismic3DDataset* rgt, std::string rgtCwtPath, const Seismic3DDataset* seismic,
		std::string seismicCwtPath, bool useSnap, bool useMedian, int lwx, int distancePower, int snapWindow, int polarity,
		float tdeb, float pasech);

template<typename RgtType, typename SeismicType>
void layerRGTInterpolatorMultiSeedDefault(const std::vector<float>& inputLayer, long dtauReference, std::vector<float>& outputLayerIso,
		std::vector<float>& outputLayerSeismic, std::vector<std::shared_ptr<FixedLayerFromDataset>>& referenceLayers,
		std::vector<RgtSeed> seeds, const Seismic3DDataset* rgt, int channelT, const Seismic3DDataset* seismic, int channelS, bool useSnap,
		bool useMedian, int lwx, int distancePower, int snapWindow, int polarity, float tdeb, float pasech);


#include "LayerRGTInterpolator.hpp"

#endif // TARUMAPP_SRC_PROCESS_LAYERRGTINTERPOLATOR_H_
