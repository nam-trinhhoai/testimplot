#ifndef SRC_SLICER_DATA_FIXEDLAYER_FIXEDRGBLAYERSFROMDATASETANDCUBEIMPLMULTI_H_
#define SRC_SLICER_DATA_FIXEDLAYER_FIXEDRGBLAYERSFROMDATASETANDCUBEIMPLMULTI_H_

#include "fixedrgblayersfromdatasetandcube.h"

class FixedRGBLayersFromDatasetAndCubeImplMulti :
		public FixedRGBLayersFromDatasetAndCube {
public:
	class Parameters : public FixedRGBLayersFromDatasetAndCube::AbstractConstructorParams {
	public:
		Parameters(QString name, QString dirPath, bool rgb1Valid);
		virtual ~Parameters();
		virtual FixedRGBLayersFromDatasetAndCube* create(QString name,
						WorkingSetManager *workingSet, const Grid3DParameter& params,
						QObject *parent = 0) override;

		virtual QString sismageName(bool* ok) const override;
		QString dirPath() const;
	private:
		QString m_dirPath;
	};

	FixedRGBLayersFromDatasetAndCubeImplMulti(QString dirPath, QString name, bool rgb1Active,
			WorkingSetManager *workingSet, const Grid3DParameter& params,
			QObject *parent = 0);
	virtual ~FixedRGBLayersFromDatasetAndCubeImplMulti();

	// ignore rgb1 for now, for compatibility
	virtual void getImageForIndex(long newIndex,
			CUDAImagePaletteHolder* redCudaBuffer, CUDAImagePaletteHolder* greenCudaBuffer,
			CUDAImagePaletteHolder* blueCudaBuffer, CUDAImagePaletteHolder* isoCudaBuffer) override;
	// use rgb1
	virtual bool getImageForIndex(long newIndex,
			QByteArray& rgbBuffer, QByteArray& isoBuffer) override;

	virtual QString getObjFile(int index) const override;

	QString dirPath() const;

	virtual QString surveyPath() const override;

	static FixedRGBLayersFromDatasetAndCube* createDataFromDatasetWithUI(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey,
			QObject *parent = 0);

	static FixedRGBLayersFromDatasetAndCube* createDataFromDatasetWithUIRgb1(QString prefix,
			WorkingSetManager *workingSet, SeismicSurvey* survey,
			QObject *parent = 0);

	static std::vector<std::shared_ptr<Parameters>> findPotentialDataRgb1(const QString& searchPath);
	static std::vector<std::shared_ptr<Parameters>> findPotentialDataRgb2(const QString& searchPath);

protected:
	virtual void setCurrentImageIndexInternal(long newIndex) override;

private:
	void loadObjectParamsFromDir(const QString& dirPath, bool useRgb1);

	QString getIsoFileFromIndex(int index) const;
	QString getRgb2FileFromIndex(int index) const;
	QString getRgb1FileFromIndex(int index) const;

	bool readRgb1(const QString& path, short* buf, long w, long h) const;

	template<typename T>
	struct CopyGDALBufToFloatBufInterleaved {
		static void run(const void* oriBuf, short* outBuf, std::size_t width, std::size_t height,
				std::size_t numBands, std::size_t offset, ImageFormats::QColorFormat colorFormat,
				GDALRasterBand* hBand);
	};
	void swapWidthHeight(const void* _oriBuf, void* _outBuf, std::size_t oriWidth,
			std::size_t oriHeight, std::size_t typeSize) const;

	QString m_dirPath;
	QMutex m_lock;
};

#endif
