#ifndef SRC_PROCESS_PATCHCOMPOSITIONPROCESS_H
#define SRC_PROCESS_PATCHCOMPOSITIONPROCESS_H

#include <vector>

#include "RgtLayerProcessUtil.h"

class Seismic3DAbstractDataset;
class FixedLayerFromDataset;
class RgtGraphLabelRead;

class PatchCompositionProcess {
public:
	PatchCompositionProcess();
	~PatchCompositionProcess();

	// seeds need to be defined in the same space as the patch volume and the output layer
	// there is currently no way to detect that.
	const std::vector<RgtSeed>& seeds() const;
	void setSeeds(const std::vector<RgtSeed>& seeds);
	void setTabIso(const std::vector<float>& tabIso);

	Seismic3DAbstractDataset* patchVolume();
	int channel() const;
	bool setPatchVolume(Seismic3DAbstractDataset* volume, int channel);

	double volumeNullValue() const;
	void setVolumeNullValue(double nullVal);

	FixedLayerFromDataset* outputLayer();
	bool setOutputLayer(FixedLayerFromDataset* layer);

	double layerNullValue() const;
	void setLayerNullValue(double nullVal);

	bool compute(int type, short seismicThreshold, std::vector<int> &vy, std::vector<int> &vz);
	void setLabelReader(RgtGraphLabelRead *reader);

private:
	/*
	template<typename PatchType>
	struct ComputeKernel {
		static bool run(PatchCompositionProcess* obj);
	};*/

	template<typename PatchType>
	struct ComputeKernel {
		static bool run(PatchCompositionProcess* obj, RgtGraphLabelRead *patchReader, int type, short seismicThreshold,
				std::vector<int> &vy, std::vector<int> &vz);
	};

	std::vector<RgtSeed> m_seeds;
	std::vector<float> m_tabIso;

	int m_channel = 0;
	Seismic3DAbstractDataset* m_patchVolume = nullptr;
	double m_volumeNullValue = 0.0f;
	FixedLayerFromDataset* m_outputLayer = nullptr;
	double m_layerNullValue = -9999.0f;
	RgtGraphLabelRead *rgtGraphLabelRead = nullptr;
};

#endif
