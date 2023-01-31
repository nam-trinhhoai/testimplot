#ifndef Seismic3DAbstractDataset_H
#define Seismic3DAbstractDataset_H

#include <QObject>
#include <QVector2D>
#include "idata.h"
#include "ifilebaseddata.h"
#include "sliceutils.h"
#include "lookuptable.h"
#include "cubeseismicaddon.h"
#include "imageformats.h"
#include "itreewidgetitemdecoratorprovider.h"
#include "Xt.h"

class CUDAImagePaletteHolder;
class Seismic3DDatasetGraphicRepFactory;
class SeismicSurvey;
class Affine2DTransformation;
class AffineTransformation;
class AbstractGraphicRep;
class IconTreeWidgetItemDecorator;

class SpectralImageCache {
public:
	SpectralImageCache();
	virtual ~SpectralImageCache();
	virtual unsigned int width() const = 0;
	virtual unsigned int height() const = 0;
	virtual unsigned int dimV() const = 0;
	virtual ImageFormats::QSampleType sampleType() const = 0;

//	virtual bool lockForRead(int timeout) const = 0;
//	virtual bool lockForWrite(int timeout) = 0;
//	virtual void unlockRead() const = 0;
//	virtual void unlockWrite() = 0;

	virtual bool copy(CUDAImagePaletteHolder *cudaImage, int channel) = 0;
};

class MonoBlockSpectralImageCache : public SpectralImageCache {
public:
	MonoBlockSpectralImageCache(unsigned int width, unsigned int height,
			unsigned int dimV, ImageFormats::QSampleType sampleType);
	virtual ~MonoBlockSpectralImageCache();
	virtual unsigned int width() const override;
	virtual unsigned int height() const override;
	virtual unsigned int dimV() const override;
	virtual ImageFormats::QSampleType sampleType() const override;

	// fastest axis are width, height and dimV
	std::vector<char>& buffer();

	virtual bool copy(CUDAImagePaletteHolder *cudaImage, int channel) override;

private:
	std::vector<char> m_buffer;
	unsigned int m_width;
	unsigned int m_height;
	unsigned int m_dimV;
	ImageFormats::QSampleType m_sampleType;
};

class ArraySpectralImageCache : public SpectralImageCache {
public:
	ArraySpectralImageCache(unsigned int width, unsigned int height,
			unsigned int dimV, ImageFormats::QSampleType sampleType);
	virtual ~ArraySpectralImageCache();
	virtual unsigned int width() const override;
	virtual unsigned int height() const override;
	virtual unsigned int dimV() const override;
	virtual ImageFormats::QSampleType sampleType() const override;

	// first vector manage dimV, second width and height; fastest being width
	// buffer ready to be copied into cuda image palette holder without transpose
	std::vector<std::vector<char>>& buffer();

	virtual bool copy(CUDAImagePaletteHolder *cudaImage, int channel) override;

private:
	std::vector<std::vector<char>> m_buffer;
	unsigned int m_width;
	unsigned int m_height;
	unsigned int m_dimV;
	ImageFormats::QSampleType m_sampleType;
};

class Volume : public IData{
	Q_OBJECT
public:
	Volume(WorkingSetManager *workingSet, QObject *parent = 0);
	virtual ~Volume();
	virtual const AffineTransformation  * const sampleTransformation() const = 0;
	virtual const Affine2DTransformation  * const ijToXYTransfo() const = 0;
	virtual const Affine2DTransformation  * const ijToInlineXlineTransfo() const = 0;
	virtual const Affine2DTransformation  * const ijToInlineXlineTransfoForInline() const = 0;
	virtual const Affine2DTransformation  * const ijToInlineXlineTransfoForXline() const = 0;

	virtual unsigned int width() const = 0;
	virtual unsigned int height() const = 0;
	virtual unsigned int depth() const = 0;
	virtual unsigned int dimV() const = 0;
	virtual ImageFormats::QSampleType sampleType() const = 0;
	virtual CubeSeismicAddon cubeSeismicAddon() const = 0;
	virtual SeismicSurvey* survey() const = 0;


	bool isCompatible(Volume* other);

	// use cache created by object.
	virtual void loadInlineXLine(CUDAImagePaletteHolder *cudaImage,
			SliceDirection dir, unsigned int z, unsigned int c=0, SpectralImageCache* cache=nullptr)=0;
	virtual void loadRandomLine(CUDAImagePaletteHolder *cudaImage,
			const QPolygon& randomLine, unsigned int c=0, SpectralImageCache* cache=nullptr)=0;

	virtual SpectralImageCache* createInlineXLineCache(SliceDirection dir) const = 0;
	virtual SpectralImageCache* createRandomCache(const QPolygon& poly) const = 0;
signals:
	void dimVChanged();
};

class Seismic3DAbstractDataset: public Volume, public IFileBasedData, public ITreeWidgetItemDecoratorProvider {
Q_OBJECT
public:
	enum CUBE_TYPE {
		Seismic, RGT, Patch
	};

	Seismic3DAbstractDataset(SeismicSurvey *survey,const QString &name,
			WorkingSetManager *workingSet, CUBE_TYPE type = CUBE_TYPE::Seismic,
			QString idPath="", QObject *parent = 0);

	virtual ~Seismic3DAbstractDataset();

	CUBE_TYPE type() const{return m_type;}

	const AffineTransformation  * const sampleTransformation() const;
	void setSampleTransformation(const AffineTransformation & transfo);

	const Affine2DTransformation  * const  ijToXYTransfo() const;

	const Affine2DTransformation  * const ijToInlineXlineTransfo() const;
	void setIJToInlineXlineTransfo(const Affine2DTransformation & transfo);

	const Affine2DTransformation  * const ijToInlineXlineTransfoForInline() const;
	void setIJToInlineXlineTransfoForInline(const Affine2DTransformation & transfo);

	const Affine2DTransformation  * const ijToInlineXlineTransfoForXline() const;
	void setIJToInlineXlineTransfoForXline(const Affine2DTransformation & transfo);

	// dimVHint is there because inri::Xt does not read dimV correctly
	// dimVHint < 1 mean no hint, dimVHint > 0 mean : use dimV = dimVHint if header dimV==1 and dimSample % dimVHint == 0
	// this can be risky because the rest of the program expect xt files to give correct dimSample and dimV
	virtual void loadFromXt(const std::string &path, int dimVHint=-1)=0;

	// cache variable ignore is is nullptr, else it should be of size dimV * height * horizontalsize
	// (width for inline, depth for xline, nb pts for random)
	// cache will be written as is from xtfile. Fastest axis is horizontalsize then height then dimV
	virtual void loadInlineXLine(CUDAImagePaletteHolder *cudaImage,
			SliceDirection dir, unsigned int z, unsigned int c=0, SpectralImageCache* cache=nullptr)=0;
	virtual void loadRandomLine(CUDAImagePaletteHolder *cudaImage,
			const QPolygon& randomLine, unsigned int c=0, SpectralImageCache* cache=nullptr)=0;

	SeismicSurvey* survey() const {
		return m_survey;
	}

	//IData
	QUuid dataID() const override;
	QString name() const override {
		return m_name;
	}

	inline unsigned int width() const {
		return m_width;
	}
	inline unsigned int height() const {
		return m_height;
	}

	inline unsigned int depth() const {
		return m_depth;
	}

	inline unsigned int dimV() const {
		return m_dimV;
	}

	ImageFormats::QSampleType sampleType() const {
		return m_sampleType;
	}

	virtual QVector2D minMax(int channel, bool forced=false);

	QRectF inlineXlineExtent() const;

	LookupTable defaultLookupTable() const;

    CubeSeismicAddon cubeSeismicAddon() const;

    static ImageFormats::QSampleType translateType(const inri::Xt::Type& type);

	virtual MonoBlockSpectralImageCache* createInlineXLineCache(SliceDirection dir) const override;
	virtual MonoBlockSpectralImageCache* createRandomCache(const QPolygon& poly) const override;

	void deleteRep();//MZR 09082021
	bool getTreeDeletionProcess() const {
		return m_TreeDeletionProcess;
	}
	void setTreeDeletionProcess(bool);
	void addRep(AbstractGraphicRep *);
	void deleteRep(AbstractGraphicRep *pRep);
	int getRepListSize();
	QList<AbstractGraphicRep*> getRepList(){return m_RepList;}

	// range lock
	bool isRangeLocked() const;
	const QVector2D& lockedRange() const;
	void lockRange(const QVector2D& range);
	void unlockRange();

	virtual bool writeRangeToFile(const QVector2D& range) = 0;
	static bool writeRangeToFile(const QVector2D& range, const std::string& xtFile);

	// ITreeWidgetItemDecoratorProvider
	virtual ITreeWidgetItemDecorator* getTreeWidgetItemDecorator() override;

signals:
   void deletedMenu();//MZR 09082021
   void rangeLockChanged();

protected:
	void initializeTransformation();
private:
	void updateIJToXYTransfo();
protected:
	void tryInitRangeLock(const std::string& xtFile);

	CUBE_TYPE m_type;

	unsigned int m_width;
	unsigned int m_height;
	unsigned int m_depth;
	unsigned int m_dimV;

	ImageFormats::QSampleType m_sampleType;

	QString m_name;
	QUuid m_uuid;

	SeismicSurvey *m_survey;

	typedef struct {
		bool initialized;
		QVector2D range;
	} MinMaxHolder;

	MinMaxHolder m_internalMinMaxCache;

	Affine2DTransformation *m_ijToInlineXline;

	//Optimizers
	Affine2DTransformation *m_ijToInlineXlineForInline;
	Affine2DTransformation *m_ijToInlineXlineForXline;

	Affine2DTransformation *m_ijToXY;

	AffineTransformation * m_sampleTransformation;

	CubeSeismicAddon m_seismicAddon;

	bool m_TreeDeletionProcess; //MZR 09082021
	QList<AbstractGraphicRep *> m_RepList;

	bool m_rangeLock;
	QVector2D m_lockedRange;

	IconTreeWidgetItemDecorator* m_treeWidgetItemDecorator;
};

Q_DECLARE_METATYPE(Seismic3DAbstractDataset*)

#endif
