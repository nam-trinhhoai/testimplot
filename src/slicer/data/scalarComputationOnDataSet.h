#ifndef ScalarComputationOnDataset_H_
#define ScalarComputationOnDataset_H_

#include "seismic3dabstractdataset.h"
#include "sliceutils.h"
#include "imageformats.h"
#include "cubeseismicaddon.h"
#include "lookuptable.h"

#include <QPolygon>
#include <QVector2D>

#include <memory>

class Seismic3DAbstractDataset;
class CUDAImagePaletteHolder;
class AffineTransformation;
class Affine2DTransformation;
class ScalarComputationOnDatasetGraphicRepFactory;

class ScalarComputationOnDataset : public Volume {
	Q_OBJECT
public:
	ScalarComputationOnDataset(Seismic3DAbstractDataset* dataset, int channel, WorkingSetManager *workingSet, QObject *parent=nullptr);
	~ScalarComputationOnDataset();

	const AffineTransformation  * const sampleTransformation() const;

	const Affine2DTransformation  * const  ijToXYTransfo() const;

	const Affine2DTransformation  * const ijToInlineXlineTransfo() const;

	const Affine2DTransformation  * const ijToInlineXlineTransfoForInline() const;

	const Affine2DTransformation  * const ijToInlineXlineTransfoForXline() const;

	void loadInlineXLine(CUDAImagePaletteHolder *cudaImage,
			SliceDirection dir, unsigned int z, unsigned int c=0, SpectralImageCache* cache=nullptr);
	void loadRandomLine(CUDAImagePaletteHolder *cudaImage,
			const QPolygon& randomLine, unsigned int c=0, SpectralImageCache* cache=nullptr);

	void computeSpectrumOnInlineXLine(SliceDirection dir, unsigned int z, short** buffer);
	void computeSpectrumOnRandom(const QPolygon& randomLine, short** buffer);

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
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

	QRectF inlineXlineExtent() const;

	LookupTable defaultLookupTable() const;

    CubeSeismicAddon cubeSeismicAddon() const;
    virtual SeismicSurvey* survey() const override;

	ArraySpectralImageCache* createInlineXLineCache(SliceDirection dir) const override;
	ArraySpectralImageCache* createRandomCache(const QPolygon& poly) const override;

	double getFrequency(long fIdx) const;
	static double getFrequencyStatic(long fIdx, double pasech, long windowSize) ;

private:
	Seismic3DAbstractDataset* m_dataset = nullptr;
	int m_channel = 0;
	int m_windowSize = 64;
	int m_hatPower = 5;

	QString m_name;
	QUuid m_uuid;
	unsigned int m_width;
	unsigned int m_height;
	unsigned int m_depth;
	unsigned int m_dimV;
	ImageFormats::QSampleType m_sampleType;

	typedef struct {
		bool initialized;
		QVector2D range;
	} MinMaxHolder;

	MinMaxHolder m_internalMinMaxCache;

	std::unique_ptr<Affine2DTransformation> m_ijToInlineXline;

	//Optimizers
	std::unique_ptr<Affine2DTransformation> m_ijToInlineXlineForInline;
	std::unique_ptr<Affine2DTransformation> m_ijToInlineXlineForXline;

	std::unique_ptr<Affine2DTransformation> m_ijToXY;

	std::unique_ptr<AffineTransformation> m_sampleTransformation;

	CubeSeismicAddon m_seismicAddon;
	ScalarComputationOnDatasetGraphicRepFactory * m_repFactory;
};

#endif
