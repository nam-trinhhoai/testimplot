#include "qglabstracttiledimage.h"

#include <QOpenGLTexture>
#include <QDebug>
#include <QFileInfo>
#include <QOpenGLPixelTransferOptions>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "texturehelper.h"
#include "gdalimagewrapper.h"
#include "qglgridutil.h"
#include "qgltile.h"

QGLAbstractTiledImage::QGLAbstractTiledImage(QObject *parent) :
		QGLAbstractImage(parent) {
	m_grid = nullptr;
}

std::vector<QGLTileCoord> QGLAbstractTiledImage::getTilesCoords(
		const QRectF &worldExtent) const {
	return m_grid->getTiles(worldExtent);
}

QGLTile* QGLAbstractTiledImage::cachedImageTile(const QGLTileCoord &coords) const {
	QMutexLocker locker(&m_cacheLock);
	CacheDataList::const_iterator it = m_cache.begin();
	for (; it != m_cache.end(); ++it) {
		if (*(*it) == coords) {
			return (*it);
		}
	}
	return nullptr;
}

void QGLAbstractTiledImage::addImageTileToCache(const QGLTileCoord &coord) {
	QRect area = coord.imageBoundingRect();
	QGLTile *tile = new QGLTile(coord, colorFormat(), sampleType());

	//Read a buffer
	tile->valid(fillTileData(area.x(), area.y(), area.width(),
					area.height(), tile->data()));

	QMutexLocker locker(&m_cacheLock);
	m_cache.push_back(tile);

	emit cachedImageInserted(tile);
}

QGLAbstractTiledImage::~QGLAbstractTiledImage() {
	qDeleteAll(m_cache);
}

