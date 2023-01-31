#ifndef QGLAbstractTiledImage_H_
#define QGLAbstractTiledImage_H_

#include <QList>
#include <QMutex>

#include "qgltilecoord.h"
#include "imageformats.h"

#include "qglabstractimage.h"

class QGLGridUtil;
class QOpenGLTexture;
class QGLTile;

typedef QList<QGLTile *> CacheDataList;

class QGLAbstractTiledImage : public QGLAbstractImage
{
	Q_OBJECT
public:
	QGLAbstractTiledImage(QObject *parent=0);
	~QGLAbstractTiledImage();

	//Tile Handling
	std::vector<QGLTileCoord> getTilesCoords(const QRectF &worldExtent) const;
	QGLTile *  cachedImageTile(const QGLTileCoord &coords) const;
	void addImageTileToCache(const QGLTileCoord &coord);

	virtual bool fillTileData(int i0, int j0, int width, int height, void *dest) const=0;

signals:
	void cachedImageInserted( QGLTile *queueItem);

protected:
	//Tiles handling
	mutable QMutex m_cacheLock;
	CacheDataList m_cache;
	QGLGridUtil * m_grid;
};


#endif /* QTLARGEIMAGEVIEWER_QGLSIMPLEIMAGE_H_ */
