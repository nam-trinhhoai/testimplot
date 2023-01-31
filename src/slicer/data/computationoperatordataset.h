#ifndef COMPUTATIONOPERATORDATASET_H
#define COMPUTATIONOPERATORDATASET_H

#include "seismic3dabstractdataset.h"
#include "sliceutils.h"
#include "imageformats.h"
#include "cubeseismicaddon.h"
#include "itreewidgetitemdecoratorprovider.h"
#include "lookuptable.h"

#include <QPolygon>
#include <QVector2D>

#include <memory>

class CUDAImagePaletteHolder;
class AffineTransformation;
class Affine2DTransformation;
class ComputationOperatorDatasetGraphicRepFactory;
class IVolumeComputationOperator;
class TextColorTreeWidgetItemDecorator;

class ComputationOperatorDataset : public Volume, public ITreeWidgetItemDecoratorProvider {
	Q_OBJECT
public:
	ComputationOperatorDataset(IVolumeComputationOperator* op, WorkingSetManager *workingSet, QObject *parent=nullptr);
	~ComputationOperatorDataset();

	const AffineTransformation  * const sampleTransformation() const;

	const Affine2DTransformation  * const  ijToXYTransfo() const;

	const Affine2DTransformation  * const ijToInlineXlineTransfo() const;

	const Affine2DTransformation  * const ijToInlineXlineTransfoForInline() const;

	const Affine2DTransformation  * const ijToInlineXlineTransfoForXline() const;

	void loadInlineXLine(CUDAImagePaletteHolder *cudaImage,
			SliceDirection dir, unsigned int z, unsigned int c=0, SpectralImageCache* cache=nullptr);
	void loadRandomLine(CUDAImagePaletteHolder *cudaImage,
			const QPolygon& randomLine, unsigned int c=0, SpectralImageCache* cache=nullptr);

	//IData
	virtual IGraphicRepFactory *graphicRepFactory();
	QUuid dataID() const override;
	QString name() const override {
		return m_name;
	}

	// ITreeWidgetItemDecoratorProvider
	virtual ITreeWidgetItemDecorator* getTreeWidgetItemDecorator() override;

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
    SeismicSurvey* survey() const;

	ArraySpectralImageCache* createInlineXLineCache(SliceDirection dir) const override;
	ArraySpectralImageCache* createRandomCache(const QPolygon& poly) const override;

	// range lock
	bool isRangeLocked() const;
	const QVector2D& lockedRange() const;
	void lockRange(const QVector2D& range);
	void unlockRange();

	IVolumeComputationOperator* computationOperator();

signals:
	void rangeLockChanged();

private:
    void setupTransforms();
    void setupInlineXlineTransfo();
    void setupInlineXlineTransfoForInline();
    void setupInlineXlineTransfoForXline();

	IVolumeComputationOperator* m_op = nullptr;

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

	mutable SeismicSurvey* m_survey = nullptr;
	CubeSeismicAddon m_seismicAddon;
	ComputationOperatorDatasetGraphicRepFactory * m_repFactory;

	bool m_rangeLock;
	QVector2D m_lockedRange;

	TextColorTreeWidgetItemDecorator* m_decorator;
};

#endif
