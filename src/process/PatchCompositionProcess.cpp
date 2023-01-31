#include "PatchCompositionProcess.h"

#include "imageformats.h"
#include "sampletypebinder.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "fixedlayerfromdataset.h"
#include "affinetransformation.h"
#include <rgtGraphLabelRead.h>

#include <algorithm>

PatchCompositionProcess::PatchCompositionProcess() {

}

PatchCompositionProcess::~PatchCompositionProcess() {

}

const std::vector<RgtSeed>& PatchCompositionProcess::seeds() const {
	return m_seeds;
}

void PatchCompositionProcess::setSeeds(const std::vector<RgtSeed>& seeds) {
	m_seeds = seeds;
}

void PatchCompositionProcess::setTabIso(const std::vector<float>& tabIso) {
	m_tabIso = tabIso;
}

Seismic3DAbstractDataset* PatchCompositionProcess::patchVolume() {
	return m_patchVolume;
}

int PatchCompositionProcess::channel() const {
	return m_channel;
}

bool PatchCompositionProcess::setPatchVolume(Seismic3DAbstractDataset* volume, int channel) {
	bool ok = false;
	if (m_outputLayer==nullptr || (volume!=nullptr && m_outputLayer->dataset()->cubeSeismicAddon().compare3DGrid(volume->cubeSeismicAddon()))) {
		m_patchVolume = volume;
		m_channel = channel;
		ok = true;
	}
	return ok;
}

double PatchCompositionProcess::volumeNullValue() const {
	return m_volumeNullValue;
}

void PatchCompositionProcess::setVolumeNullValue(double nullVal) {
	m_volumeNullValue = nullVal;
}

FixedLayerFromDataset* PatchCompositionProcess::outputLayer() {
	return m_outputLayer;
}

bool PatchCompositionProcess::setOutputLayer(FixedLayerFromDataset* layer) {
	bool ok = false;
	if (layer->keys().contains(FixedLayerFromDataset::ISOCHRONE) &&
			(m_patchVolume==nullptr || (layer!=nullptr &&
					layer->dataset()->cubeSeismicAddon().compare3DGrid(m_patchVolume->cubeSeismicAddon())))) {
		m_outputLayer = layer;
		ok = true;
	}
	return ok;
}

double PatchCompositionProcess::layerNullValue() const {
	return m_layerNullValue;
}

void PatchCompositionProcess::setLayerNullValue(double nullVal) {
	m_layerNullValue = nullVal;
}

void PatchCompositionProcess::setLabelReader(RgtGraphLabelRead *reader)
{
	rgtGraphLabelRead = reader;
}

bool PatchCompositionProcess::compute(int type, short seismicThreshold, std::vector<int> &vy, std::vector<int> &vz) {
	if (m_outputLayer==nullptr || m_patchVolume==nullptr ||
			dynamic_cast<Seismic3DDataset*>(m_patchVolume)==nullptr
			// || !m_outputLayer->keys().contains(FixedLayerFromDataset::ISOCHRONE)
			) {
		return false;
	}

	SampleTypeBinder binder(m_patchVolume->sampleType());
	return binder.bind<ComputeKernel>(this, this->rgtGraphLabelRead, type, seismicThreshold, vy, vz);
}


template<typename PatchType>
bool PatchCompositionProcess::ComputeKernel<PatchType>::run(PatchCompositionProcess* obj, RgtGraphLabelRead *patchReader, int type, short seismicThreshold,
		std::vector<int> &vy, std::vector<int> &vz) {

	if ( patchReader == nullptr ) return false;
	const std::vector<RgtSeed>& seeds = obj->seeds();
	Seismic3DDataset* volume = dynamic_cast<Seismic3DDataset*>(obj->patchVolume());
	FixedLayerFromDataset* layer = obj->outputLayer();

	std::vector<int>xSeed;
	std::vector<int>ySeed;
	std::vector<int>zSeed;
	xSeed.resize(seeds.size());
	ySeed.resize(seeds.size());
	zSeed.resize(seeds.size());
	for (int i=0; i<seeds.size(); i++)
	{
		xSeed[i] = seeds[i].x;
		ySeed[i] = seeds[i].y;
		zSeed[i] = seeds[i].z;
	}

	std::vector<float> iso0(layer->getNbProfiles()*layer->getNbTraces());
	for (int n=0; n<iso0.size(); n++) if ( iso0[n] == 0 ) iso0[n] = -9999;
	layer->readProperty(iso0.data(), FixedLayerFromDataset::ISOCHRONE);
	patchReader->setSeismicThreshold(seismicThreshold);
	std::vector<float> iso;
	if ( type == 0 )
		iso = patchReader->getTabIso(xSeed, ySeed, zSeed, iso0);
	else if ( type == 1 )
		iso = patchReader->getNeighborTabIso(iso0);
	else if ( type == 2 )
		iso = patchReader->eraseArea(vy, vz, iso0);
	layer->writeProperty(iso.data(), FixedLayerFromDataset::ISOCHRONE);

	return true;
}


/*
template<typename PatchType>
bool PatchCompositionProcess::ComputeKernel<PatchType>::run(PatchCompositionProcess* obj) {
	const std::vector<RgtSeed>& seeds = obj->seeds();
	Seismic3DDataset* volume = dynamic_cast<Seismic3DDataset*>(obj->patchVolume());
	FixedLayerFromDataset* layer = obj->outputLayer();
	int channel = obj->channel();
	double layerNullValue = obj->layerNullValue();
	double volumeNullValue = obj->volumeNullValue();
	const AffineTransformation* sampleTransfo = volume->sampleTransformation();

	std::vector<int> patchIds;
	patchIds.reserve(seeds.size());

	std::vector<PatchType> valTab;
	valTab.resize(volume->dimV());
	fprintf(stderr, "seeds: %d\n", seeds.size());
	for (const RgtSeed& seed : seeds) {
		std::vector<int> offsets = {0, 1, -1, 2, -2};
		int offsetIdx = 0;
		bool patchNotFound = true;
		while (offsetIdx<offsets.size() && patchNotFound) {
			int x = seed.x + offsets[offsetIdx];
			if (x>=0 && x<volume->height()) {
				volume->readSubTraceAndSwap(valTab.data(), x, x+1, seed.y, seed.z);
				PatchType val = valTab[channel];
				if (val!=volumeNullValue) {
					std::vector<int>::const_iterator it = std::find(patchIds.begin(), patchIds.end(), val);
					if (it==patchIds.end()) {
						patchIds.push_back(val);
					}
					patchNotFound = false;
				}
			}
			offsetIdx++;
		}
	}

	std::vector<float> iso;
	iso.resize(static_cast<std::size_t>(layer->getNbProfiles()) * layer->getNbTraces());
	layer->readProperty(iso.data(), FixedLayerFromDataset::ISOCHRONE);

	std::size_t height = volume->height();
	std::size_t dimV = volume->dimV();

	std::vector<PatchType> patchSection;
	patchSection.resize(dimV * height * volume->width());
	for (std::size_t i=0; i<layer->getNbProfiles(); i++) {
		volume->readTraceBlockAndSwap(patchSection.data(), 0, layer->getNbTraces(), i);
#pragma omp parallel for schedule(dynamic)
		for (std::size_t j=0; j<layer->getNbTraces(); j++) {
			std::size_t idx = j + i * layer->getNbTraces();
			if (iso[idx]==layerNullValue) {
				std::size_t patchIdx = 0;
				std::size_t volIdx = 0;
				bool notFound = true;
				while (notFound && patchIdx<patchIds.size()) {
					volIdx = 0;
					while (notFound && volIdx<height) {
						PatchType val = patchSection[(volIdx + j * height) * dimV + channel];
						notFound = val==volumeNullValue || val!=patchIds[patchIdx];
						if (notFound) {
							volIdx++;
						}
					}
					if (notFound) {
						patchIdx++;
					}
				}
				if (!notFound) {
					double outVal;
					sampleTransfo->direct(volIdx, outVal);
					iso[idx] = outVal;
				}
			}
		}
	}
	layer->writeProperty(iso.data(), FixedLayerFromDataset::ISOCHRONE);

	return true;
}
*/

