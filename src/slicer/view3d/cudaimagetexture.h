#ifndef CUDAIMAGETEXTURE_H
#define CUDAIMAGETEXTURE_H

#include <Qt3DRender/QTexture>
#include "imageformats.h"

class CudaImageTexture : public Qt3DRender::QTexture2D
{
    Q_OBJECT
	 Q_PROPERTY(QByteArray data READ data WRITE setData NOTIFY dataChanged)
public:
    explicit CudaImageTexture(ImageFormats::QColorFormat colorFormat, ImageFormats::QSampleType sampleType,int width, int height,Qt3DCore::QNode *parent = nullptr);
    ~CudaImageTexture();

    QByteArray data() const;
    void setData(const QByteArray &data);

    ImageFormats::QSampleType sampleType();

signals:
    void dataChanged();

private:
    QByteArray m_data;
    ImageFormats::QSampleType m_sampleType;
    ImageFormats::QColorFormat m_colorFormat;
    int m_width;
    int m_height;

};

#endif // COLORTABLETEXTURE_H
