#ifndef LayerDataset_H
#define LayerDataset_H

#include <QMutex>
#include "seismic3dabstractdataset.h"

#include "RgtLayerProcessUtil.h"

class LayerDatasetGraphicRepFactory;
class Seismic3DAbstractDataset;

class ToAnalyse2Process;
class ToAnalyse3Process;
class ToAnalyse4Process;

class LayerDataset: public Seismic3DAbstractDataset {
Q_OBJECT
public:
	LayerDataset(SeismicSurvey *survey,const QString &name, WorkingSetManager *workingSet,
			Seismic3DAbstractDataset *datasetS, Seismic3DAbstractDataset *datasetT,
			CUBE_TYPE type = CUBE_TYPE::Seismic, QObject *parent = 0);
	virtual ~LayerDataset();

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();

	void readInlineBlock(void *output, int z0, int z1);
	void loadFromXt(const std::string &path);
	void loadInlineXLine(CUDAImagePaletteHolder *cudaImage, SliceDirection dir,
			unsigned int z);

	std::string path() const {
		return m_path;
	}

	size_t headerLength() const {
		return m_headerLength;
	}
	virtual QVector2D minMax(bool forced=false) override;

	void setSeeds(const std::vector<RgtSeed>& seeds);
	bool getPolarity() const;
	void setPolarity(bool polarity);
	int getDistancePower() const;
	void setDistancePower(int);
	bool getUseSnap() const;
	void setUseSnap(bool val);
	int getSnapWindow() const;
	void setSnapWindow(int);
	bool getUseMedian() const;
	void setUseMedian(bool val);
	int getLWXMedianFilter() const;
	void setLWXMedianFilter(int lwx);
	int getWindowSize() const ;
	void setWindowSize(int windowSize = 64);
	float getHatPower() const;
	void setHatPower(float hatPower);

	void setFreqMax(int freqMax );
	void setFreqMin(int freqMin );
	void setFreqStep(int freqStep ) ;

	void setW(int w );
	void setShift(int shift );

	int getMethod() const;
	void setMethod(int method );

	Seismic3DAbstractDataset*& getDatasetS() {
		return m_datasetS;
	}

	int getFreqMax() const {
		return m_freqMax;
	}

	int getFreqMin() const {
		return m_freqMin;
	}

	int getFreqStep() const {
		return m_freqStep;
	}

	void computeProcess();

private:
	void releaseContent();

private:
	LayerDatasetGraphicRepFactory *m_repFactory;
	Seismic3DAbstractDataset* m_datasetS;
	Seismic3DAbstractDataset* m_datasetT;
	QMutex m_lock;
	FILE *m_currentFile;
	size_t m_headerLength;
	std::string m_path;

	int m_distancePower = 8;
	bool m_polarity = true;
	bool m_useSnap = false;
	int m_snapWindow = 3;
	bool m_useMedian = false;
	int m_lwx_medianFilter = 11;
	int m_windowSize = 64;
	float m_hatPower = 5;
	int m_freqMin = 20;
	int m_freqMax = 150;
	int m_freqStep = 2;
	int m_w = 10;
	int m_shift = 0;
	int m_method = 0;

	std::vector<RgtSeed> m_seeds;

	ToAnalyse2Process* m_layerSpectrumProcess = nullptr;
	ToAnalyse3Process* m_morletProcess = nullptr;
	ToAnalyse4Process* m_gccProcess = nullptr;
};
Q_DECLARE_METATYPE(LayerDataset*)
#endif
