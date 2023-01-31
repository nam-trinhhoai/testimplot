#ifndef QGLAbstractImage_H
#define QGLAbstractImage_H

#include <QVector2D>
#include <QMatrix4x4>
#include "lookuptable.h"
#include "ipaletteholder.h"
#include "igeorefimage.h"
#include "imageformats.h"

class QOpenGLTexture;

//A full image read from GDAL: as a proof of concept. Should be abstracted to a more higher level interface
class QGLAbstractImage : public QObject,public IPaletteHolder,public IGeorefImage
{
	Q_OBJECT

	//IGeorefImage mapping
	Q_PROPERTY(int width READ width CONSTANT)
	Q_PROPERTY(int height READ height CONSTANT)

	Q_PROPERTY(LookupTable lookupTable READ lookupTable WRITE setLookupTable NOTIFY lookupTableChanged)
	Q_PROPERTY(QVector2D rangeRatio READ rangeRatio  NOTIFY rangeChanged)
	Q_PROPERTY(float opacity READ opacity WRITE setOpacity NOTIFY opacityChanged)

public:
	QGLAbstractImage(QObject *parent=0);
	~QGLAbstractImage();

	QRectF worldExtent() const override;

	void bindLUTTexture(unsigned int unit);
	void releaseLUTTexture(unsigned int unit);

	float opacity() const;
	LookupTable lookupTable() const;

	virtual QVector2D rangeRatio() override;
	virtual QVector2D range() override;
	virtual QVector2D dataRange() override;

	virtual ImageFormats::QColorFormat colorFormat() const;
	virtual ImageFormats::QSampleType sampleType() const;
	virtual bool value(double worldX, double worldY,int &i, int &j,double &value) const;

	bool hasNoDataValue() const
	{
		return m_hasNodataValue;
	};
	float noDataValue() const
	{
		return m_noDataValue;
	};
signals:
	void opacityChanged();
	void rangeChanged();
	void lookupTableChanged();

public slots:
	void setOpacity(float value);
	void setLookupTable(const LookupTable & table);
	void setRange(const QVector2D &range);
protected:
	virtual QVector2D computeRange() const=0;
private:
	void initRange();
	void updateRangeRatio();
protected:
	QOpenGLTexture* m_colorMapTexture;
	LookupTable m_lookupTable;
	bool m_needColorTableReload;

	float m_opacity;

	float m_noDataValue;
	bool m_hasNodataValue;

	QVector2D m_rangeRatio;
	QVector2D m_range;
	QVector2D m_dataRange;
	bool m_dataRangeComputed;

	ImageFormats::QColorFormat m_colorFormat;
	ImageFormats::QSampleType m_samplType;
};


#endif /* QTLARGEIMAGEVIEWER_QGLSIMPLEIMAGE_H_ */
