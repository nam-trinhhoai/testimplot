/*
 * FixedRGBLayersFromDatasetAndCube.h
 *
 *  Created on: Oct. 02, 2020
 *      Author: l0483271
 */

#ifndef SRC_SLICER_DATA_FIXEDLAYER_FIXEDRGBLAYERSFROMDATASETANDCUBE_H_
#define SRC_SLICER_DATA_FIXEDLAYER_FIXEDRGBLAYERSFROMDATASETANDCUBE_H_

#include <QObject>
#include <QString>
#include <memory>
#include <vector>
#include <list>
#include <QByteArray>
#include <QTimer>
#include <QMutex>

#include "idata.h"
#include "cubeseismicaddon.h"
#include "stackabledata.h"
#include "surfacemeshcacheutils.h"
#include "cudargbinterleavedimage.h"
#include "cudaimagepaletteholder.h"
#include "cpuimagepaletteholder.h"
#include "imageformats.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include "qhistogram.h"
#include "isochronprovider.h"
// #include <seismic3dabstractdataset.h>

#include <memory>

class IGraphicRepFactory;
class Seismic3DAbstractDataset;
class SeismicSurvey;
class GDALRasterBand;

class FixedRGBLayersFromDatasetAndCube : public IData, public StackableData, public IsoChronProvider {
Q_OBJECT
private:
	QString m_dataType = "spectrum";
	void *m_option = nullptr;
	int m_optionAttribut = 0;
	QString m_dataSetName = "";
	// Seismic3DAbstractDataset *m_dataSet3D = nullptr;
public:
	typedef struct Grid3DParameter {
		unsigned int width;
		unsigned int depth;
		unsigned int heightFor3D; // only use for 3d
		std::shared_ptr<AffineTransformation> sampleTransformation;
		std::shared_ptr<Affine2DTransformation> ijToXYTransfo;
		std::shared_ptr<Affine2DTransformation> ijToInlineXlineTransfoForInline;
		std::shared_ptr<Affine2DTransformation> ijToInlineXlineTransfoForXline;
		CubeSeismicAddon cubeSeismicAddon;
	} Grid3DParameter;
	void setOption(void *val) { m_option = val; }
	void *getOption() { return m_option; }
	void setOptionAttribut(int val) { m_optionAttribut = val; }
	int getOptionAttribut() { return m_optionAttribut; }
	void setDataSetName(QString val) { m_dataSetName = val; }
	QString getDataSetName() { return m_dataSetName; }
	QString getName() { return m_name; }
	virtual void updateDataAttribut() {};

	enum Mode { READ, CACHE };

	typedef struct SurfaceCache{
		QByteArray rgb; // interleave RGB
		QByteArray iso;
		QVector2D redRange;
		QVector2D greenRange;
		QVector2D blueRange;
		QHistogram redHistogram;
		QHistogram greenHistogram;
		QHistogram blueHistogram;

		SurfaceMeshCache meshCache;

	} SurfaceCache;

	// This class only allow to manipulate more easily params to construct child object of FixedRGBLayersFromDatasetAndCube
	class AbstractConstructorParams {
	public:
		AbstractConstructorParams(QString name, bool rgb1Valid);
		virtual ~AbstractConstructorParams();
		virtual FixedRGBLayersFromDatasetAndCube* create(QString name,
				WorkingSetManager *workingSet, const Grid3DParameter& params,
				QObject *parent = 0) = 0;

		virtual QString sismageName(bool* ok) const = 0;
		QString name() const;
		bool rgb1Valid() const;
	private:
		QString m_name;
		bool m_rgb1Valid;
		QString m_dataType = "spectrum";
	};

	// cudaBuffer need to be a float RGBD planar stack
//	FixedRGBLayersFromDatasetAndCube(QString cubeIso, QString name,
//			WorkingSetManager *workingSet, const Grid3DParameter& params,
//			QObject *parent = 0);
//	FixedRGBLayersFromDatasetAndCube(QString rgb2Path, QString rgb1Path, QString name,
//			WorkingSetManager *workingSet, const Grid3DParameter& params,
//			QObject *parent = 0);
	FixedRGBLayersFromDatasetAndCube(QString name,
			WorkingSetManager *workingSet, const Grid3DParameter& params,
			QObject *parent = 0);
	virtual ~FixedRGBLayersFromDatasetAndCube();

	unsigned int width() const;
	unsigned int depth() const;
	unsigned int getNbProfiles() const;
	unsigned int getNbTraces() const;
	unsigned int heightFor3D() const; // only use for 3d

	float getStepSample();
	float getOriginSample();

	IsoSurfaceBuffer getIsoBuffer()override;

	void play(int interval, int coef,bool looping);

	// buffer access
	std::size_t numLayers();
	const QList<QString>& layers() const;
	//use rgb1
	long currentImageIndex() const;
	void setCurrentImageIndex(long newIndex);
	// ignore rgb1 for now, for compatibility
	virtual void getImageForIndex(long newIndex,
			CUDAImagePaletteHolder* redCudaBuffer, CUDAImagePaletteHolder* greenCudaBuffer,
			CUDAImagePaletteHolder* blueCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer) = 0;
	// use rgb1
	virtual bool getImageForIndex(long newIndex,
			QByteArray& rgbBuffer, QByteArray& isoBuffer) = 0;

	// void setDataSet3D(Seismic3DAbstractDataset *dataSet3D) { m_dataSet3D = dataSet3D; }
	// Seismic3DAbstractDataset *getDataSet3D() { return m_dataSet3D; }

	//IData
	virtual IGraphicRepFactory* graphicRepFactory() override;
	QUuid dataID() const override;
	QString name() const override;


	bool modePlay()
	{
		return m_modePlay;
	}
	CUDARGBInterleavedImage* image() {
		return m_currentRGB.get();
	}

	CPUImagePaletteHolder* isoSurfaceHolder() {
		return m_currentIso.get();
	}

	virtual QColor getHorizonColor()
	{
		return m_color;
	}

	virtual void setHorizonColor(QColor color)
	{
		if (m_color!=color)
		{
			m_color = color;
			emit colorChanged(m_color);
		}
	}

	virtual QString dirPath() const { return ""; }


	virtual void *getHorizonShape() { return nullptr; }

	virtual QString getIsoFileFromIndex(int index) { return ""; }

	virtual void updateTextColor(QString item) { }

	virtual void setGeotimeGraphicsView(void *val) {  }


	bool isIsoInT() const {
		return m_isIsoInT;
	}

	long isoStep() const {
		return m_isoStep;
	}

	long isoOrigin() const {
		return m_isoOrigin;
	}


	long getCurrentTime() const{
		return m_isoOrigin + m_isoStep * m_currentImageIndex;
	}


//	static FixedRGBLayersFromDatasetAndCube* createDataFromDatasetWithUI(QString prefix,
//			WorkingSetManager *workingSet, SeismicSurvey* survey,
//			QObject *parent = 0);
//
//	static FixedRGBLayersFromDatasetAndCube* createDataFromDatasetWithUIRgb1(QString prefix,
//			WorkingSetManager *workingSet, SeismicSurvey* survey,
//			QObject *parent = 0);

	long getIsoStep() const {
		return m_isoStep;
	}

	long getIsoOrigin() const {
		return m_isoOrigin;
	}

	virtual QString getObjFile(int index) const = 0;
	QString getCurrentObjFile() const;
	SurfaceMeshCache* getMeshCache(int index);
	bool isIndexCache(int index);

	int getSimplifyMeshSteps()const;
	int getCompressionMesh()const;


	Mode mode() const;
	void moveToReadMode();
	bool moveToCacheMode(long firstIso, long lastIso, long isoStep);

	// only valid if mode is "Cache"
	long cacheFirstIndex() const;
	long cacheLastIndex() const;
	long cacheStepIndex() const;

	bool useRgb1() const;
	//QString dirPath() const;
//	QString rgb2Path() const;
//	QString rgb1Path() const;
	CubeSeismicAddon cubeSeismicAddon() const;
	const AffineTransformation* sampleTransformation() const;
	const Affine2DTransformation* ijToXYTransfo() const;
	const Affine2DTransformation* ijToInlineXlineTransfoForInline() const;
	const Affine2DTransformation* ijToInlineXlineTransfoForXline() const;

	virtual QString surveyPath() const = 0;

	virtual std::vector<StackType> stackTypes() const override;
	virtual std::shared_ptr<AbstractStack> stack(StackType type) override;

	// use a default mesh step to compute mesh cache size
	long long cacheLayerMemoryCost() const;

	// HSV minimum value
	bool isMinimumValueActive() const;
	void setMinimumValueActive(bool active);
	float minimumValue() const;
	void setMinimumValue(float minimumValue);

	virtual void setRedIndex(int value) { }
	virtual void setGreenIndex(int value) { }
	virtual void setBlueIndex(int value) { }
	virtual int getRedIndex() { return 0; }
	virtual int getGreenIndex() { return 0; }
	virtual int getBlueIndex() { return 0; }
	virtual void setRGBIndexes(int r, int g, int b) {}
	virtual void setGrayFreqIndexes(int val) {}

	virtual int getNbreSpectrumFreq() { return 0; }
	virtual QString getLabelFromPosition(int index) { return ""; }



	template<typename InputType>
	static bool checkValidity(const QByteArray& vect, std::size_t expectedSize);

	static QString extractListAndJoin(QStringList list, long beg, long end, QString joinStr);

	static Grid3DParameter createGrid3DParameter(const QString& datasetPath, SeismicSurvey* survey, bool* ok);
	static Grid3DParameter createGrid3DParameterFromHorizon(const QString& horizonPath, SeismicSurvey* survey, bool* ok);

	static FixedRGBLayersFromDatasetAndCube* createDataFromDatasetWithUI(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
			QString dataType = "spectrum",
			QObject *parent = 0);

	static std::vector<FixedRGBLayersFromDatasetAndCube*> createHorizonIsoDataFromDatasetWithUI(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
					QString dataType = "spectrum",
					QObject *parent = 0);

	static std::vector<FixedRGBLayersFromDatasetAndCube*> createDataFromDataset(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
			QString dataType = "spectrum",
			QObject *parent = 0);

	static std::vector<FixedRGBLayersFromDatasetAndCube*> createDataFreeHorizonFromDataset(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
			QString dataType = "spectrum",
			QObject *parent = 0);

	static std::vector<FixedRGBLayersFromDatasetAndCube*> createDataFreeHorizonFromDatasetWithUI(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey, bool useRgb1,
					QString dataType = "spectrum",
					QObject *parent = 0);

	static FixedRGBLayersFromDatasetAndCube* createDataFreeHorizonFromHorizonPath(QString horizonPath, WorkingSetManager *workingSet, SeismicSurvey* survey,
			QObject *parent = 0);

	virtual bool isInlineXLineDisplay() { return true; }

	virtual QString propertyPanelType() { return "default"; }

	bool isInitialized() const;
	void initialize();
	// virtual void buildContextMenu(QMenu *menu) override;


signals:
	void frequencyChanged();

public slots:
	void setSimplifyMeshSteps(int steps);
	void setCompressionMesh( int compress);
	void nextCurrentIndex();

signals:
	void modeChanged();
	void initProgressBar(int min, int max, int val);
	void valueProgressBarChanged(int value);
	void endProgressBar();
	void currentIndexChanged(long index);

	void simplifyMeshStepsChanged(int value);
	void compressionMeshChanged(int value);

	void minimumValueActivated(bool value);
	void minimumValueChanged(float value);
	void colorChanged(QColor color);

protected:
	void initLayersList();
	virtual void setCurrentImageIndexInternal(long newIndex) = 0;

	bool m_useRgb1;
	std::size_t m_numLayers = 0;
	long m_isoStep;
	long m_isoOrigin;

	std::unique_ptr<CPUImagePaletteHolder> m_currentIso = nullptr;
	std::unique_ptr<CUDARGBInterleavedImage> m_currentRGB = nullptr;
	long m_currentImageIndex = -1; // index for m_selectedLayersKeys

	std::list<SurfaceCache> m_cacheList;

private:
	//void loadObjectParamsFromDir(const QString& dirPath, bool useRgb1);
	//void loadIsoAndRgb(QString rgb2, QString rgb1="");

//	QString getIsoFileFromIndex(int index) const;
//	QString getRgb2FileFromIndex(int index) const;
//	QString getRgb1FileFromIndex(int index) const;

//	template<typename ValType>
//	void swap(ValType& val);

//	bool readRgb1(const QString& path, short* buf, long w, long h) const;

//	template<typename T>
//	struct CopyGDALBufToFloatBufInterleaved {
//		static void run(const void* oriBuf, short* outBuf, std::size_t width, std::size_t height,
//				std::size_t numBands, std::size_t offset, ImageFormats::QColorFormat colorFormat,
//				GDALRasterBand* hBand);
//	};
	/*void copyGDALBufToFloatBufInterleaved(void* oriBuf, short* outBuf, std::size_t width, std::size_t height,
			std::size_t numBands, std::size_t offset, ImageFormats::QColorFormat colorFormat,
			ImageFormats::QSampleType sampleType, GDALRasterBand* hBand);*/

//	void swapWidthHeight(const void* _oriBuf, void* _outBuf, std::size_t oriWidth,
//			std::size_t oriHeight, std::size_t typeSize) const;

	std::unique_ptr<IGraphicRepFactory> m_repFactory;

	unsigned int m_width;
	unsigned int m_depth;
	unsigned int m_heightFor3D; // only use for 3d
	std::unique_ptr<AffineTransformation> m_sampleTransformation;
	std::unique_ptr<Affine2DTransformation> m_ijToXYTransfo;
	std::unique_ptr<Affine2DTransformation> m_ijToInlineXlineTransfoForInline;
	std::unique_ptr<Affine2DTransformation> m_ijToInlineXlineTransfoForXline;
	CubeSeismicAddon m_cubeSeismicAddon;
	QUuid m_uuid;
	QString m_name;

	QList<QString> m_layers;

	bool m_useMinimumValue = false;
	float m_minimumValue = 0.0f; // from HSV standpoint

//	FILE* m_fRgb2;
//	QString m_rgb2Path;
//	FILE* m_fRgb1;
//	QString m_rgb1Path;
//	QString m_dirPath;
	bool m_isIsoInT = true;

//	QMutex m_lock;
//	QMutex m_lockRgb1;
//	QMutex m_lockRgb2;
	QMutex m_lockRead;
	QMutex m_lockNextRead;

	long m_nextIndex = -1;// if val<0 means not set

	Mode m_mode;
	long m_cacheFirstIso;
	long m_cacheLastIso;
	long m_cacheStepIso;
	long m_cacheFirstIndex;
	long m_cacheLastIndex;
	long m_cacheStepIndex;

	int m_simplifyMeshSteps;
	int m_compressionMesh;

	int m_defaultSimplifyMeshSteps; // for cacheLayerMemoryCost

	// this should not be used except for the getter/setter functions of color
	// the child class can redefine the getter/setter so use these functions instead of m_color
	QColor m_color = Qt::red;
	bool m_modePlay;
	bool m_loop = false;
	int m_incr = 1;
	int m_coef;
	QTimer *m_timerRefresh;

	bool m_init = false;
};

class FixedRGBLayersFromDatasetAndCubeStack : public AbstractRangeStack {
	Q_OBJECT
public:
	FixedRGBLayersFromDatasetAndCubeStack(FixedRGBLayersFromDatasetAndCube* data);
	virtual ~FixedRGBLayersFromDatasetAndCubeStack();

	virtual long stackCount() const override;
	virtual long stackIndex() const override;

	virtual QVector2D stackRange() const override;
	virtual double stackStep() const override;

	virtual double stackValueFromIndex(long index) const override;
	virtual long stackIndexFromValue(double value) const override;

public slots:
	virtual void setStackIndex(long stackIndex) override;

private slots:
	void indexChangedFromData(long stackIndex);

private:
	FixedRGBLayersFromDatasetAndCube* m_data;
};

#include "fixedrgblayersfromdatasetandcube.hpp"

#endif /* SRC_SLICER_DATA_FIXEDLAYER_FIXEDRGBLAYERSFROMDATASETANDCUBE_H_ */
