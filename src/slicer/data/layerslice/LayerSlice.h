/*
 * LayerSlice.h
 *
 *  Created on: Jun 15, 2020
 *      Author: l0222891
 */

#ifndef SRC_SLICER_DATA_LAYERSLICE_LAYERSLICE_H_
#define SRC_SLICER_DATA_LAYERSLICE_LAYERSLICE_H_

#include <QObject>
#include <QMutex>
#include <memory>
#include "idata.h"

#include "LayerSpectrumProcess.h"
#include "RgtLayerProcessUtil.h"
#include "MorletProcess.h"
#include "GradientMultiScaleProcess.h"
#include "KohonenLayerProcess.h"
#include "AttributProcess.h"
#include "AnisotropyProcess.h"
#include "lookuptable.h"
#include "isochronprovider.h"
#include "itreewidgetitemdecoratorprovider.h"


class TextColorTreeWidgetItemDecorator;
class Seismic3DDataset;
class IGraphicRepFactory;
class CUDAImagePaletteHolder;

class ToAnalyse2Process;
class ToAnalyse3Process;
class ToAnalyse4Process;
class ToAnalyse5Process;
class FixedLayerFromDataset;
class KohonenLayerProcess;
class AbstractGraphicRep;

// Care, these enum elements are used as vector indexes for LayerSpectrumDialog
typedef enum
{
	eComputeMethd_Morlet =0,
	eComputeMethd_Spectrum = 1,
	eComputeMethd_Gcc = 2,
	eComputeMethd_TMAP = 3,
	eComputeMethd_Mean = 4,
	eComputeMethd_Anisotropy = 5,
	eComputeMethd_RGB = 6,
	eComputeMethd_Max
} eComputeMethod;

class LayerSlice: public IData, public IsoChronProvider, public ITreeWidgetItemDecoratorProvider {
Q_OBJECT
public:
	typedef struct PaletteParameters {
		QVector2D range;
		LookupTable lookupTable;
	} PaletteParameters;

	LayerSlice(WorkingSetManager *workingSet,
			Seismic3DDataset *seismic, int channelS,
			Seismic3DDataset *rgt, int channelT,
			int iDataType = 0, // MZR 09072021
			QObject *parent = 0);
	virtual ~LayerSlice();

	uint extractionWindow() const;
	int getComputationType() const;
	void deleteRgt();

	void setExtractionWindow(uint w);

	int currentPosition() const;
	void setSlicePosition(int pos);
	QString getCurrentLabel() const;

	unsigned int width() const;
	unsigned int height() const;
	unsigned int depth() const;

	QUuid seismicID() const;
	QUuid rgtID() const;

	CUDAImagePaletteHolder* isoSurfaceHolder() {
		return m_isoSurfaceHolder.get();
	}
	CUDAImagePaletteHolder* image() {
		return m_image.get();
	}
	Seismic3DDataset* seismic() const {
		return m_datasetS;
	}

	QVector2D rgtMinMax();

	IsoSurfaceBuffer getIsoBuffer()override;

	//IData
	virtual IGraphicRepFactory* graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override;

	// ITreeWidgetItemDecoratorProvider
	virtual ITreeWidgetItemDecorator* getTreeWidgetItemDecorator() override;

	void lockPalette(const QString& label, const PaletteParameters& params);
	void unlockPalette(const QString& label);
	std::pair<bool, PaletteParameters> getLockedPalette(const QString& label) const;

public:
signals:
	void extractionWindowChanged(unsigned int size);
	void RGTIsoValueChanged(int pos);
	void computationFinished(int nbOutputSlices);
	void deletedMenu();
	void deleteRgtLayer();// 17082021

public:
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
	int getGccOffset() const;
	void setGccOffset(int gccOffset);
	float getHatPower() const;
	void setHatPower(float hatPower);
	long getDTauReference() const;
	void setDTauReference(long dtau);

	void setFreqMax(int freqMax );
	void setFreqMin(int freqMin );
	void setFreqStep(int freqStep ) ;

	void setW(int w );
	void setShift(int shift );

	int getMethod() const;
	void setMethod(int method );
	void setType(int type);

	Seismic3DDataset*& getDatasetS() {
		return m_datasetS;
	}

	int getChannelS() {
		return m_channelS;
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

	void computeProcess(LayerSpectrumDialog *layerspectrumdialog);

	const float* getModuleData(unsigned int);
	unsigned int getNbOutputSlices() const;
	bool isModuleComputed() const;

	void setConstrainLayer(FixedLayerFromDataset* constrain, QString propName);
	void setReferenceLayers(const std::vector<std::shared_ptr<FixedLayerFromDataset>>& references,
			QString isoName, QString rgtName);

	double getFrequency(long fIdx) const;
	static double getFrequencyStatic(long fIdx, double pasech, long windowSize) ;

	int getTmapExampleSize() const;
	void setTmapExampleSize(int val);

	int getTmapSize() const;
	void setTmapSize(int val);

	int getTmapExampleStep() const;
	void setTmapExampleStep(int val);

	void setAttributDatasets(const QList<std::pair<Seismic3DDataset*, int>>& datasets);
	void setDatasetsAndAngles(const QList<std::tuple<Seismic3DDataset*, int, float>>& datasetsAndAngles);

	QString getLabelFromPosition(int val) const;

	int getSimplifyMeshSteps() const;
	void setSimplifyMeshSteps(int steps);

	int getCompressionMesh() const;
	void setCompressionMesh(int compress);

	void deleteRep();
	void setProcessDeletion(bool);
	bool ProcessDeletion() const;
private:
	void releaseContent();
	void loadSlice(unsigned int z);
	void loadSlice(CUDAImagePaletteHolder *image,
			unsigned int extractionWindow, unsigned int z);
	void loadSlice(CUDAImagePaletteHolder *isoSurfaceImage,
			CUDAImagePaletteHolder *image, unsigned int extractionWindow,
			unsigned int z);
private:
	unsigned int m_extractionWindow;
	unsigned int m_currentSlice;

	std::unique_ptr<IGraphicRepFactory> m_repFactory;

	Seismic3DDataset* m_datasetS;
	int m_channelS;
	Seismic3DDataset* m_datasetT;
	int m_channelT;
//	QMutex m_lock;
//	FILE *m_currentFile;
//	size_t m_headerLength;
//	std::string m_path;

	std::unique_ptr<CUDAImagePaletteHolder> m_isoSurfaceHolder;
	std::unique_ptr<CUDAImagePaletteHolder> m_image;

	int m_distancePower = 8;
	bool m_polarity = true;
	bool m_useSnap = false;
	int m_snapWindow = 3;
	bool m_useMedian = false;
	int m_lwx_medianFilter = 11;
	int m_windowSize = 64;
	int m_gccOffset = 7;
	float m_hatPower = 5;
	int m_freqMin = 20;
	int m_freqMax = 150;
	int m_freqStep = 2;
	int m_w = 10;
	int m_shift = 0;
	int m_method = 0;
	long m_dtauReference = 0;
	int m_type = 0; // gcc or gcc mean

	int m_tmapExampleSize = 10;
	int m_tmapSize = 33;
	int m_tmapExampleStep = 20;
	QString m_Name;
	int m_Comptype;

	std::vector<RgtSeed> m_seeds;
	QList<std::pair<Seismic3DDataset*, int>> m_datasets;
	QList<std::tuple<Seismic3DDataset*, int, float>> m_datasetsAndAngles;

	std::unique_ptr<ToAnalyse2Process> m_layerSpectrumProcess = nullptr;
	std::unique_ptr<ToAnalyse3Process> m_morletProcess = nullptr;
	std::unique_ptr<ToAnalyse4Process> m_gccProcess = nullptr;
	std::unique_ptr<KohonenLayerProcess> m_tmapProcess = nullptr;
	std::unique_ptr<ToAnalyse5Process> m_attributProcess = nullptr;
	std::unique_ptr<AnisotropyAbstractProcess> m_anisotropyProcess = nullptr;
	FixedLayerFromDataset* m_constrainLayer = nullptr; // data given to use in the process
	std::vector<std::shared_ptr<FixedLayerFromDataset>> m_referenceLayer;
	QString m_referenceIsoName;
	QString m_referenceTauName;
	QString m_constrainIsoName;
	bool m_TreeDeletionProcess=false;

	int m_simplifyMeshSteps;
	int m_compressionMesh;
	QList<QMetaObject::Connection> m_conn;
	std::map<QString, PaletteParameters> m_cachedPaletteParameters;

	TextColorTreeWidgetItemDecorator* m_decorator;
};

#endif /* SRC_SLICER_DATA_LAYERSLICE_LAYERSLICE_H_ */
