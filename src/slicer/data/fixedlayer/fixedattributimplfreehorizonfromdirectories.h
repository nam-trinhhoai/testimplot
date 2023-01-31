#ifndef SRC_SLICER_DATA_FIXEDLAYER_FIXEDATTRIBUTIMPLFREEHORIZONFROMDIRECTORIES_H_
#define SRC_SLICER_DATA_FIXEDLAYER_FIXEDATTRIBUTIMPLFREEHORIZONFROMDIRECTORIES_H_

#include <vector>
#include <QColor>
#include <QMenu>
#include <QAction>

#include "GraphEditor_PolyLineShape.h"
#include "fixedrgblayersfromdatasetandcube.h"

class FixedAttributImplFreeHorizonFromDirectories :
		public FixedRGBLayersFromDatasetAndCube {

private:
	class ParamSpectrum
	{
	public:
		int f1 = 2;
		int f2 = 4;
		int f3 = 6;
		int nfreq = 33;
	};
	class ParamGcc
	{
		public:
		int f1 = 2;
		int f2 = 3;
		int f3 = 4;
	};
public:
	class Parameters : public FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams {
	public:
		Parameters(QString name, QString dirPath, bool rgb1Valid, QString dataType, std::vector<QString> seismicName);
		virtual ~Parameters();
		virtual FixedRGBLayersFromDatasetAndCube* create(QString name,
						WorkingSetManager *workingSet, const Grid3DParameter& params,
						QObject *parent = 0) override;

		virtual QString sismageName(bool* ok) const override;
		QString dirPath() const;
		QString m_dataType = "spectrum";
		QString m_seismicName = "";
	private:
		QString m_dirPath;
		QString m_dirName = "";
		std::vector<QString> m_dataSetNames;
		// QString
	};

	/*
	FixedAttributImplFromDirectories(QString dirPath, QString name, bool rgb1Active,
			WorkingSetManager *workingSet, const Grid3DParameter& params,
			const QString &dataType,
			QObject *parent = 0);
			*/

	FixedAttributImplFreeHorizonFromDirectories(QString dirPath, QString dirName, std::vector<QString> seismicName, WorkingSetManager *workingSet,
			const Grid3DParameter& params, QObject *parent = 0);

	virtual ~FixedAttributImplFreeHorizonFromDirectories();

	// ignore rgb1 for now, for compatibility
	virtual void getImageForIndex(long newIndex,
			CUDAImagePaletteHolder* redCudaBuffer, CUDAImagePaletteHolder* greenCudaBuffer,
			CUDAImagePaletteHolder* blueCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer) override;
	// use rgb1
	virtual bool getImageForIndex(long newIndex,
			QByteArray& rgbBuffer, QByteArray& isoBuffer) override;

	virtual QString getObjFile(int index) const override;

 	QString dirPath() const;
 	QString dirName() { return m_dirName; }

	virtual QString surveyPath() const override;

	static FixedRGBLayersFromDatasetAndCube* createDataFromDatasetWithUI(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey,
			QObject *parent = 0);

	static FixedRGBLayersFromDatasetAndCube* createDataFromDatasetWithUIRgb1(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey,
			QObject *parent = 0);

	static std::vector<std::shared_ptr<FixedAttributImplFreeHorizonFromDirectories::Parameters>> findPotentialData(const QString& searchPath);
	static std::shared_ptr<FixedAttributImplFreeHorizonFromDirectories::Parameters> findData(const QString& horizonPath);


	static std::vector<std::shared_ptr<Parameters>> findPotentialDataRgb1(const QString& searchPath);
	static std::vector<std::shared_ptr<Parameters>> findPotentialDataRgb2(const QString& searchPath, const QString& dataType);
	void updateDataAttribut();
	std::vector<QString> getDataSetNames() { return m_dataSetNames; }
	QColor getHorizonColor();
	void setHorizonColor(QColor col);
	// GraphEditor_PolyLineShape *getHorizonShape();
	// void setDataSet3D(Seismic3DAbstractDataset *val);
	virtual bool isInlineXLineDisplay();

	QString propertyPanelType() override { return "3freq"; }

	void setRedIndex(int value);
	void setGreenIndex(int value);
	void setBlueIndex(int value);
	int getNbreSpectrumFreq() override;

	int getRedIndex() override ;
	int getGreenIndex() override ;

	int getBlueIndex() override ;
	void setRGBIndexes(int r, int g, int b) override ;
	void setGrayFreqIndexes(int idx) override;
	QString getLabelFromPosition(int index) override;


/*
signals:
	void frequencyChanged();
	*/

protected:
	virtual void setCurrentImageIndexInternal(long newIndex) override;

private:
	QString m_dataSetPath = "";
	QString getSpectrumDataSet();


	void loadObjectParamsFromDir(const QString& dirPath, const QString& dirName, const QString& seismicName);

	QString getIsoFileFromIndex(int index);
	QString getSpectrumFileFromIndex(int index);
	QString getGccFileFromIndex(int index);
	QString getMeanFileFromIndex(int index);
	QString getAttributFileFromIndex(int index);
	QString readAttributFromFile(int index, void *buff, long size);


	// QString getRgb2FileFromIndex(int index) const;
	// QString getRgb1FileFromIndex(int index) const;

	bool readRgb1(const QString& path, short* buf, long w, long h) const;

	template<typename T>
	struct CopyGDALBufToFloatBufInterleaved {
		static void run(const void* oriBuf, short* outBuf, std::size_t width, std::size_t height,
				std::size_t numBands, std::size_t offset, ImageFormats::QColorFormat colorFormat,
				GDALRasterBand* hBand);
	};
	void swapWidthHeight(const void* _oriBuf, void* _outBuf, std::size_t oriWidth,
			std::size_t oriHeight, std::size_t typeSize) const;

	static QString getRgtFilename(QString fullPath);
	static QString getSeismicFilename(QString fullPath);
	static QString getFilenameFromPath(QString fullPath);
	static QString getPathUp(QString fullPath);
	static QString getHorizonName(QString fullPath);
	static QString getSeismicNameFromFile(QString filename);
	float getPasEch();
	float getFrequency(int index);


	// void actionMenuCreate();

	ParamSpectrum m_paramSpectrum;
	ParamGcc m_paramGcc;

	std::vector<QString> m_dataSetNames;
 	QString m_dirPath = "";
	QString m_dirName = "";
	QString m_seismicName = "";
	QMutex m_lock;
	std::vector<QString> m_dirNames;
	QString m_dataType = "spectrum";

	QString m_gccNames;
	QString m_isoNames;
	QString m_meanNames;
	QString m_spectrumNames;
	// Seismic3DAbstractDataset *m_dataSet3D= nullptr;
	WorkingSetManager *m_workingSetManager = nullptr;

	int m_nbreSpectrumFreq = -1;
	float m_pasEch = -1.0;


	bool m_redSet=false;
	bool m_greenSet = false;
	bool m_blueSet= false;

	QString m_attributName="";

	// GraphEditor_PolyLineShape *m_polylineShape = nullptr;
	// QMenu *m_itemMenu = nullptr;
	// QAction *m_actionColor = nullptr;
	// QAction *m_actionProperties = nullptr;
	// QAction *m_actionLocation = nullptr;

	/*
	public slots:
	void trt_changeColor();
	void trt_properties();
	void trt_location();
	*/

};

#endif
