/*
 * FixedRGBLayersFromDatasetAndCube.h
 *
 *  Created on: Oct. 02, 2020
 *      Author: l0483271
 */

#ifndef SRC_SLICER_DATA_FIXEDLAYER_FIXEDLAYERSFROMDATASETANDCUBE_H_
#define SRC_SLICER_DATA_FIXEDLAYER_FIXEDLAYERSFROMDATASETANDCUBE_H_

#include <QObject>
#include <QString>
#include <memory>
#include <vector>
#include <list>
#include <memory>
#include <QColor>

#include "idata.h"
#include "cubeseismicaddon.h"
#include "stackabledata.h"
#include "qhistogram.h"
#include "lookuptable.h"
#include "surfacemeshcacheutils.h"
#include "cpuimagepaletteholder.h"
#include "isochronprovider.h"

#include <QMutex>

class IGraphicRepFactory;
class Seismic3DAbstractDataset;
class CUDAImagePaletteHolder;
class CPUImagePaletteHolder;
class AffineTransformation;
class Affine2DTransformation;
class SeismicSurvey;

class FixedLayersFromDatasetAndCube : public IData, public StackableData, public IsoChronProvider {
Q_OBJECT
public:
/*
	typedef struct Grid3DParameter {
		unsigned int width;
		unsigned int depth;
		unsigned int heightFor3D; // only use for 3d
		const AffineTransformation* sampleTransformation;
		const Affine2DTransformation* ijToXYTransfo;
		const Affine2DTransformation* ijToInlineXlineTransfoForInline;
		const Affine2DTransformation* ijToInlineXlineTransfoForXline;
		CubeSeismicAddon cubeSeismicAddon;
	} Grid3DParameter;
	*/
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

	enum Mode { READ, CACHE };

	typedef struct SurfaceCache{
		QByteArray attr;
		QByteArray iso;

		QVector2D attrRange;
		QHistogram attrHistogram;
		SurfaceMeshCache meshCache;
	} SurfaceCache;

	// cudaBuffer need to be a float RGBD planar stack
	FixedLayersFromDatasetAndCube(QString cubeIso, QString name,
			WorkingSetManager *workingSet, const Grid3DParameter& params,
			QObject *parent = 0,
			bool valide = true);
	virtual ~FixedLayersFromDatasetAndCube();

	unsigned int width() const;
	unsigned int depth() const;
	unsigned int getNbProfiles() const;
	unsigned int getNbTraces() const;
	unsigned int heightFor3D() const; // only use for 3d

	float getStepSample();
	float getOriginSample();

	void play(int interval, int coef,bool looping);

	// buffer access
	std::size_t numLayers();
	const QList<QString>& layers() const;
	//use rgb1
	long currentImageIndex() const;
	virtual void setCurrentImageIndex(long newIndex);
	// ignore rgb1 for now, for compatibility
	virtual void getImageForIndex(long newIndex,
			CUDAImagePaletteHolder* attrCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer);
	// use rgb1
	virtual bool getImageForIndex(long newIndex,
			QByteArray& attrBuffer, QByteArray& isoBuffer);

	//IData
	virtual IGraphicRepFactory* graphicRepFactory() override;
	QUuid dataID() const override;
	QString name() const override;

	bool modePlay()
	{
		return m_modePlay;
	}

	IsoSurfaceBuffer getIsoBuffer()override;

	CPUImagePaletteHolder* image() {
		return m_currentAttr.get();
	}

	CPUImagePaletteHolder* isoSurfaceHolder() {
		return m_currentIso.get();
	}

	bool isIsoInT() const {
		return m_isIsoInT;
	}

	long isoStep() const {
		return m_isoStep;
	}

	long isoOrigin() const {
		return m_isoOrigin;
	}

	static FixedLayersFromDatasetAndCube* createDataFromDatasetWithUI(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey,
			QObject *parent = 0);

	long getIsoStep() const {
		return m_isoStep;
	}

	long getIsoOrigin() const {
		return m_isoOrigin;
	}

	long getCurrentTime() const{
			return m_isoOrigin + m_isoStep * m_currentImageIndex;
		}


	virtual QString getObjFile(int index) const;
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

	QString attributePath() const;
	CubeSeismicAddon cubeSeismicAddon() const;
	const AffineTransformation* sampleTransformation() const;
	const Affine2DTransformation* ijToXYTransfo() const;
	const Affine2DTransformation* ijToInlineXlineTransfoForInline() const;
	const Affine2DTransformation* ijToInlineXlineTransfoForXline() const;

	virtual std::vector<StackType> stackTypes() const override;
	virtual std::shared_ptr<AbstractStack> stack(StackType type) override;

	template<typename InputType>
	static bool checkValidity(const QByteArray& vect, std::size_t expectedSize);
	static QString extractListAndJoin(QStringList list, long beg, long end, QString joinStr);

	void lockPalette(const QVector2D& range, const LookupTable& lookupTable);
	void unlockPalette();
	bool isPaletteLocked() const;
	const QVector2D& lockedRange() const;
	const LookupTable& lockedLookupTable() const;

	// use a default mesh step to compute mesh cache size
	long long cacheLayerMemoryCost() const;
	static Grid3DParameter createGrid3DParameter(
		const QString& datasetPath, SeismicSurvey* survey, bool* ok);
	static Grid3DParameter createGrid3DParameterFromHorizon(
		const QString& horizonPath, SeismicSurvey* survey, bool* ok);

	virtual QColor getHorizonColor() { return m_color; }
	virtual QString dirPath() { return m_attrPath; }
	virtual void setHorizonColor(QColor col) {
		if (m_color!=col) {
			m_color = col;
			emit colorChanged(m_color);
		}
	};
	virtual QString getIsoFileFromIndex(int index) const;

	virtual bool enableSlicePropertyPanel() { return true; }
	virtual QString propertyPanelType() { return "default"; }
	virtual bool enableScaleSlider() { return false; }
	virtual int getNbreGccScales() { return 1; }
	virtual void setGccIndex(int value) { }

	bool isInitialized() const;
	void initialize();

	QString sectionToolTip() const;
	void setSectionToolTip(const QString&);

protected:
	std::vector<QString> m_dirNames;
	std::size_t m_numLayers;
	long m_isoOrigin;
	long m_isoStep;
	int m_dataType = 0;
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
	bool m_modePlay;
	int m_coef;
	std::unique_ptr<IGraphicRepFactory> m_repFactory;
	std::unique_ptr<CPUImagePaletteHolder> m_currentIso = nullptr;
	std::unique_ptr<CPUImagePaletteHolder> m_currentAttr = nullptr;
	long m_currentImageIndex = -1; // index for m_selectedLayersKeys


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

	void colorChanged(QColor color);

private:
	virtual void loadIsoAndAttribute(QString attribute);
	virtual void setCurrentImageIndexInternal(long newIndex);

	template<typename ValType>
	void swap(ValType& val);


	FILE* m_f;
	QString m_attrPath;
	bool m_isIsoInT = true;
	// do not use m_color except for getter/setter, use getter/setter instead to allow the child class to manage the color
	QColor m_color = Qt::white;
	bool m_init = false;
	QString m_sectionToolTip;

protected:
	QMutex m_lock;
	QMutex m_lockFile;
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
	std::list<SurfaceCache> m_cacheList;

	bool m_isLockUsed = false;
	QVector2D m_lockedRange;
	LookupTable m_lockedLookupTable;

	int m_simplifyMeshSteps;
	int m_compressionMesh;

	int m_defaultSimplifyMeshSteps;

	bool m_loop = false;
	int m_incr = 1;
	QTimer *m_timerRefresh;
	QList<QString> m_layers;

	QString getMeanFileFromIndex(int index) const;
};

class FixedLayersFromDatasetAndCubeStack : public AbstractRangeStack {
	Q_OBJECT
public:
	FixedLayersFromDatasetAndCubeStack(FixedLayersFromDatasetAndCube* data);
	virtual ~FixedLayersFromDatasetAndCubeStack();

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
	FixedLayersFromDatasetAndCube* m_data;
};

#include "fixedlayersfromdatasetandcube.hpp"

#endif /* SRC_SLICER_DATA_FIXEDLAYER_FIXEDRGBLAYERSFROMDATASETANDCUBE_H_ */
