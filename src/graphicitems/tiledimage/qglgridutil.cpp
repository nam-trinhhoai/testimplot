#include "qglgridutil.h"
#include <cmath>
#include <algorithm>
#include <iostream>

#include "igeorefimage.h"

int QGLGridUtil::TILE_SIZE = 1024;

QGLGridUtil::QGLGridUtil(IGeorefImage *provider, int width, int height,
		QObject *parent) :
		QObject(parent) {
	m_width = width;
	m_height = height;

	int numTileX = width / TILE_SIZE + 1;
	int numTileY = height / TILE_SIZE + 1;

	m_tiles.reserve(numTileX * numTileY);

	for (int j = 0; j < numTileY; j++) {
		for (int i = 0; i < numTileX; i++) {
			int i0 = i * TILE_SIZE;
			int j0 = j * TILE_SIZE;
			int w = TILE_SIZE;
			int h = TILE_SIZE;
			if (i0 + w > width)
				w = width - TILE_SIZE;
			if (j0 + h > height)
				h = height - TILE_SIZE;

			std::vector<QVertex2D> glCoords(4);
			double x, y;
			provider->imageToWorld(i0, j0 + h, x, y);
			glCoords[0] = QVertex2D(QVector2D(x, y), QVector2D(0, 1));

			provider->imageToWorld(i0, j0, x, y);
			glCoords[1] = QVertex2D(QVector2D(x, y), QVector2D(0, 0));

			provider->imageToWorld(i0 + w, j0 + h, x, y);
			glCoords[2] = QVertex2D(QVector2D(x, y), QVector2D(1, 1));

			provider->imageToWorld(i0 + w, j0, x, y);
			glCoords[3] = QVertex2D(QVector2D(x, y), QVector2D(1, 0));

			float xmin = x, ymin = y, xmax = x, ymax = y;
			for (int i = 0; i < 3; i++) {
				xmin = std::min(xmin, glCoords[i].position.x());
				ymin = std::min(ymin, glCoords[i].position.y());

				xmax = std::max(xmax, glCoords[i].position.x());
				ymax = std::max(ymax, glCoords[i].position.y());
			}
			m_tiles.push_back(
					QGLTileCoord(QRect(i0, j0, w, h),
							QRectF(xmin, ymin, xmax - xmin, ymax - ymin),
							glCoords));
		}
	}
}

std::vector<QGLTileCoord> QGLGridUtil::getTiles(
		const QRectF &worldExtent) const {
	std::vector<QGLTileCoord> result;
	for (const QGLTileCoord tc : m_tiles) {
		if (worldExtent.intersects(tc.worldBoundingRect()))
			result.push_back(tc);
	}
	return result;
}

std::vector<QRect> QGLGridUtil::getITiles(int pos) const {

	int numTileX = m_width / TILE_SIZE + 1;
	int numTileY = m_height / TILE_SIZE + 1;

	std::vector<QRect> result;
	for (int j = 0; j < numTileY; j++) {
		int j0 = j * TILE_SIZE;
		int h = TILE_SIZE;
		if (j0 + h > m_height)
			h = m_height - TILE_SIZE;
		for (int i = 0; i < numTileX; i++) {
			int i0 = i * TILE_SIZE;
			int w = TILE_SIZE;
			if (i0 + w > m_width)
				w = m_width - TILE_SIZE;

			if (i0 + w < pos)
				continue;

			result.push_back(QRect(i0, j0, w, h));
			break;
		}
	}
	return result;
}

std::vector<QRect> QGLGridUtil::getJTiles(int pos) const {

	int numTileX = m_width / TILE_SIZE + 1;
	int numTileY = m_height / TILE_SIZE + 1;

	std::vector<QRect> result;
	for (int j = 0; j < numTileY; j++) {
		int j0 = j * TILE_SIZE;
		int h = TILE_SIZE;
		if (j0 + h > m_height)
			h = m_height - TILE_SIZE;

		if (j0 + h < pos)
			continue;
		for (int i = 0; i < numTileX; i++) {
			int i0 = i * TILE_SIZE;
			int w = TILE_SIZE;

			if (i0 + w > m_width)
				w = m_width - TILE_SIZE;

			result.push_back(QRect(i0, j0, w, h));
		}
		break;
	}
	return result;
}

QRect QGLGridUtil::getTile(int px, int py) const {

	int numTileX = m_width / TILE_SIZE + 1;
	int numTileY = m_height / TILE_SIZE + 1;

	std::vector<QRect> result;
	for (int j = 0; j < numTileY; j++) {
		int j0 = j * TILE_SIZE;
		int h = TILE_SIZE;
		if (j0 + h > m_height)
			h = m_height - TILE_SIZE;

		if (j0 + h < py)
			continue;
		for (int i = 0; i < numTileX; i++) {
			int i0 = i * TILE_SIZE;
			int w = TILE_SIZE;
			if (i0 + w > m_width)
				w = m_width - TILE_SIZE;

			if (i0 + w < px)
				continue;

			return QRect(i0, j0, w, h);
		}
	}
	return QRect();
}

void QGLGridUtil::dump() {
	for (const QGLTileCoord tc : m_tiles)
		qDebug() << tc.imageBoundingRect() << tc.worldBoundingRect();
}

