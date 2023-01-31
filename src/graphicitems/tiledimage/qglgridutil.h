#ifndef QGLGridUtil_H
#define QGLGridUtil_H

#include <QRectF>
#include <QObject>
#include "qgltilecoord.h"

class IGeorefImage;

class QGLGridUtil :public QObject{
	Q_OBJECT
public:
	static int TILE_SIZE;

	QGLGridUtil(IGeorefImage * provider,int width, int height,QObject *parent=0);
	std::vector<QGLTileCoord> getTiles(const QRectF &worldExtent) const;

	QRect getTile(int px, int py) const;
	std::vector<QRect> getITiles(int pos) const;
	std::vector<QRect> getJTiles(int pos) const;

	void dump();
private:
	std::vector<QGLTileCoord> m_tiles;
	int m_width;
	int m_height;
};

#endif
