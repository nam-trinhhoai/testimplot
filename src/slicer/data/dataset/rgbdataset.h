#ifndef Data_RgbDataset_H
#define Data_RgbDataset_H

#include "idata.h"
#include "sliceutils.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include <QPolygon>
#include <cmath>

class Volume;
class RgbDatasetGraphicRepFactory;
class CUDAImagePaletteHolder;
class SpectralImageCache;

class RgbDataset : public IData {
	Q_OBJECT
public:
	enum Mode { NONE = -4, TRANSPARENT = -3, OPAQUE = -2, OTHER = -1};

	~RgbDataset();

	QString name() const override {
		return m_name;
	}
	virtual QUuid dataID() const override;
	virtual IGraphicRepFactory* graphicRepFactory() override;

	static RgbDataset* createRgbDataset(const QString name, Volume* r, int channelR,
			Volume* g, int channelG,
			Volume* b, int channelB,
			Volume* a=nullptr, int channelA=0,
			WorkingSetManager *workingSet=nullptr, QObject *parent = 0);

	const AffineTransformation  * const sampleTransformation() const;
	const Affine2DTransformation  * const  ijToXYTransfo() const;
	const Affine2DTransformation  * const ijToInlineXlineTransfo() const;
	const Affine2DTransformation  * const ijToInlineXlineTransfoForInline() const;
	const Affine2DTransformation  * const ijToInlineXlineTransfoForXline() const;

	virtual void loadInlineXLine(CUDAImagePaletteHolder *redImage, CUDAImagePaletteHolder *greenImage,
			CUDAImagePaletteHolder *blueImage, CUDAImagePaletteHolder *alphaImage, SliceDirection dir,
			unsigned int z, SpectralImageCache* redCache=nullptr, SpectralImageCache* greenCache=nullptr,
			SpectralImageCache* blueCache=nullptr, SpectralImageCache* alphaCache=nullptr);
	virtual void loadRandomLine(CUDAImagePaletteHolder *redImage, CUDAImagePaletteHolder *greenImage,
			CUDAImagePaletteHolder *blueImage, CUDAImagePaletteHolder *alphaImage,
			const QPolygon& randomLine, SpectralImageCache* redCache=nullptr, SpectralImageCache* greenCache=nullptr,
			SpectralImageCache* blueCache=nullptr, SpectralImageCache* alphaCache=nullptr);

	inline unsigned int width() const {
		return m_width;
	}
	inline unsigned int height() const {
		return m_height;
	}

	inline unsigned int depth() const {
		return m_depth;
	}

	QRectF inlineXlineExtent() const;

	SampleUnit sampleUnit() const;

	Volume* red() const;
	int channelRed() const;
	void setChannelRed(int val);
	Volume* green() const;
	int channelGreen() const;
	void setChannelGreen(int val);
	Volume* blue() const;
	int channelBlue() const;
	void setChannelBlue(int val);
	Volume* alpha() const;
	int channelAlpha() const;
	void setChannelAlpha(int val);

	Mode alphaMode() const;
	void setAlphaMode(Mode mode);
	float constantAlpha() const;
	void setConstantAlpha(float alpha);
	float radiusAlpha() const;
	void setRadiusAlpha(float radius);

signals:
	void alphaModeChanged();
	void constantAlphaChanged();
	void radiusAlphaChanged();
	void redChannelChanged();
	void greenChannelChanged();
	void blueChannelChanged();
	void alphaChannelChanged();

private:
	RgbDataset(const QString name, Volume* r, int channelR,
			Volume* g, int channelG,
			Volume* b, int channelB,
			Volume* a=nullptr, int channelA=0,
			WorkingSetManager *workingSet=nullptr, QObject *parent = 0);

	Volume* m_red;
	int m_channelRed;
	Volume* m_green;
	int m_channelGreen;
	Volume* m_blue;
	int m_channelBlue;
	Volume* m_alpha;
	int m_channelAlpha;

	Mode m_alphaMode;
	float m_constantAlpha = 1;
	float m_radiusAlpha = 0; // 0 to sqrt(6)

	QString m_name;
	QUuid m_uuid;
	RgbDatasetGraphicRepFactory *m_repFactory;

	unsigned int m_width, m_height, m_depth;

	Affine2DTransformation m_ijToInlineXline;

	//Optimizers
	Affine2DTransformation m_ijToInlineXlineForInline;
	Affine2DTransformation m_ijToInlineXlineForXline;

	Affine2DTransformation m_ijToXY;

	AffineTransformation m_sampleTransformation;

	SampleUnit m_sampleUnit;
};

#endif
