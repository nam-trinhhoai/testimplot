#include "colortabletexture.h"
#include "colortableregistry.h"
#include <Qt3DRender/QAbstractTextureImage>
#include <Qt3DRender/QTextureImageDataGenerator>
#include <array>

namespace {

class SkinImageGenerator : public Qt3DRender::QTextureImageDataGenerator
{
public:
    explicit SkinImageGenerator(const LookupTable &lookupTable)
        : m_lookupTable(lookupTable)
    {}

    ~SkinImageGenerator() {}

    QT3D_FUNCTOR(SkinImageGenerator)

    Qt3DRender::QTextureImageDataPtr operator ()() override
    {
        Qt3DRender::QTextureImageDataPtr textureData =  Qt3DRender::QTextureImageDataPtr::create();
        textureData->setTarget(QOpenGLTexture::Target1D);
        textureData->setFormat(QOpenGLTexture::RGBA8_UNorm);
        textureData->setPixelFormat(QOpenGLTexture::RGBA);
        textureData->setPixelType(QOpenGLTexture::UInt8);
        textureData->setWidth(m_lookupTable.size());
        textureData->setHeight(1);

        QByteArray rawData;
        rawData.resize(m_lookupTable.size() * 4);
        for (int i = 0, m = m_lookupTable.size(); i < m; ++i) {
            const std::array<int, 4> colors = m_lookupTable.getColors(i);
            rawData[4 * i] = (unsigned char)colors[0];
            rawData[4 * i + 1] = (unsigned char)colors[1];
            rawData[4 * i + 2] = (unsigned char)colors[2];
            rawData[4 * i + 3] = (unsigned char)colors[3];
//            qDebug() << Q_FUNC_INFO << rawData[4 * i] <<  rawData[4 * i + 1] << rawData[4 * i + 2] << rawData[4 * i + 3];
        }
        textureData->setData(rawData, 1, false);
        return textureData;
    }

    bool operator ==(const Qt3DRender::QTextureImageDataGenerator &other) const override
    {
        const SkinImageGenerator *gen = functor_cast<SkinImageGenerator>(&other);
        if (gen == this)
            return true;
        return false;
    }

private:
    LookupTable m_lookupTable;
};
using ColorTableImageGeneratorPtr = QSharedPointer<SkinImageGenerator>;

class ColorTableImage : public Qt3DRender::QAbstractTextureImage
{
    Q_OBJECT
public:

    explicit ColorTableImage(const ColorTableImageGeneratorPtr &generator,
                             Qt3DCore::QNode *parent = nullptr)
        : Qt3DRender::QAbstractTextureImage(parent)
        , m_generator(generator)
    {}

    ~ColorTableImage()
    {}

protected:
    // QAbstractTextureImage interface
    Qt3DRender::QTextureImageDataGeneratorPtr dataGenerator() const override
    {
        return m_generator;
    }

private:
    ColorTableImageGeneratorPtr m_generator;
};


} // anonymous

ColorTableTexture::ColorTableTexture(Qt3DCore::QNode *parent)
    : Qt3DRender::QTexture1D(parent)
{
    setMinificationFilter(Qt3DRender::QTexture1D::Nearest);
    setMagnificationFilter(Qt3DRender::QTexture1D::Nearest);
    setGenerateMipMaps(false);
    setWrapMode(Qt3DRender::QTextureWrapMode(Qt3DRender::QTextureWrapMode::ClampToEdge));
    setLookupTable(ColorTableRegistry::DEFAULT());
}

ColorTableTexture::~ColorTableTexture()
{
}

void ColorTableTexture::setLookupTable(const LookupTable &table)
{
    m_lookupTable = table;

    const QVector<Qt3DRender::QAbstractTextureImage *> images = textureImages();
    for (Qt3DRender::QAbstractTextureImage *img : images) {
        removeTextureImage(img);
        img->deleteLater();
    }

    setWidth(table.size());
    setHeight(1);
    setFormat(Qt3DRender::QTexture1D::RGBA8_UNorm);

    auto generator = ColorTableImageGeneratorPtr::create(m_lookupTable);
    ColorTableImage *img = new ColorTableImage(generator, this);
    addTextureImage(img);

    emit lookupTableChanged();
}

LookupTable ColorTableTexture::lookupTable() const
{
    return m_lookupTable;
}
#include "colortabletexture.moc"
