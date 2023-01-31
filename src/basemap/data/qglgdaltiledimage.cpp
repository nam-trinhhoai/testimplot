#include "qglgdaltiledimage.h"

#include <QOpenGLTexture>
#include "texturehelper.h"
#include "gdalimagewrapper.h"
#include "qglgridutil.h"
#include "qgltile.h"

QGLGDALTiledImage::QGLGDALTiledImage(QObject *parent) :
QGLAbstractTiledImage(parent) {
	m_internalImage = new GDALImageWrapper(this);
}

QGLGDALTiledImage::~QGLGDALTiledImage() {
	close();
}

void QGLGDALTiledImage::close() {
	m_internalImage->close();
}

int QGLGDALTiledImage::width() const {
	return m_internalImage->width();
}
int QGLGDALTiledImage::height() const {
	return m_internalImage->height();
}

void QGLGDALTiledImage::worldToImage(double worldX, double worldY,
		double &imageX, double &imageY) const {
	m_internalImage->worldToImage(worldX, worldY, imageX, imageY);
}
void QGLGDALTiledImage::imageToWorld(double imageX, double imageY,
		double &worldX, double &worldY) const {
	m_internalImage->imageToWorld(imageX, imageY, worldX, worldY);
}

QMatrix4x4 QGLGDALTiledImage::imageToWorldTransformation() const {
	return m_internalImage->imageToWorldTransformation();
}

bool QGLGDALTiledImage::open(const QString &imageFilePath) {
	close();
	m_internalImage->close();
	if (!m_internalImage->open(imageFilePath))
		return false;

	m_grid = new QGLGridUtil(this, width(), height(), this);

	m_noDataValue = m_internalImage->noDataValue(m_hasNodataValue);
	m_colorFormat = m_internalImage->colorFormat();
	m_samplType = m_internalImage->sampleType();
	return true;
}
bool QGLGDALTiledImage::fillTileData(int i0, int j0, int width, int height,
		void *dest) const {
	return m_internalImage->readData(i0, j0, width, height, dest,
			m_internalImage->numBands());
}

QVector2D QGLGDALTiledImage::computeRange() const {
	return m_internalImage->computeRange();
}

QHistogram QGLGDALTiledImage::computeHistogram(const QVector2D &range,
		int nBuckets)  {
	if (range == m_cachedHisto.range())
		return m_cachedHisto;
	return m_internalImage->computeHistogram(range, nBuckets);
}

bool QGLGDALTiledImage::valueAt(int i, int j, double &value) const {
	QRect tile = m_grid->getTile(i, j);
	QMutexLocker locker(&m_cacheLock);
	CacheDataList::const_iterator it = m_cache.begin();
	for (; it != m_cache.end(); ++it) {
		QGLTile *const t = *it;
		if (t->coords().imageBoundingRect() == tile) {
			return TextureHelper::valueAt(t->data(), i - tile.x(), j - tile.y(),
					t->width(), sampleType(), value);
		}
	}
	return false;

}
void QGLGDALTiledImage::valuesAlongJ(int j, bool *valid, double *values) const {
	std::vector<QRect> tiles = m_grid->getJTiles(j);
	QMutexLocker locker(&m_cacheLock);
	CacheDataList::const_iterator it = m_cache.begin();
	for (int j = 0; j < width(); j++)
		valid[j] = false;

	for (; it != m_cache.end(); ++it) {
		QGLTile *const t = *it;
		for (const QRect tile : tiles) {
			if (t->coords().imageBoundingRect() == tile) {
				TextureHelper::valuesAlongJ(t->data(), tile.x(), j - tile.y(),
						valid, values, t->width(), t->height(), sampleType());
			}
		}
	}
}
void QGLGDALTiledImage::valuesAlongI(int i, bool *valid, double *values) const {
	std::vector<QRect> tiles = m_grid->getITiles(i);
	QMutexLocker locker(&m_cacheLock);
	CacheDataList::const_iterator it = m_cache.begin();
	for (int j = 0; j < height(); j++)
		valid[j] = false;

	for (; it != m_cache.end(); ++it) {
		QGLTile *const t = *it;
		for (const QRect tile : tiles) {
			if (t->coords().imageBoundingRect() == tile) {
				TextureHelper::valuesAlongI(t->data(), i - tile.x(), tile.y(),
						valid, values, t->width(), t->height(), sampleType());
			}
		}
	}
}
