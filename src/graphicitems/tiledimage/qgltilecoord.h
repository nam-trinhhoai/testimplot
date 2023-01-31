#ifndef QGLTileCoord_H
#define QGLTileCoord_H

#include <vector>
#include <QRect>
#include "qvertex2D.h"

class QGLTileCoord {
public:
	QGLTileCoord();
	QGLTileCoord(const QRect &imagePosition, const QRectF & worldPosition, const  std::vector<QVertex2D> &glCoords);
	QGLTileCoord(const QGLTileCoord &tc);

	QGLTileCoord & operator= ( const QGLTileCoord & val );
	 bool operator==(const QGLTileCoord  & rhs) const;

	virtual ~QGLTileCoord();

	 inline const QRect imageBoundingRect() const{
	      return m_imagePosition;
	 }

	 inline const QRectF worldBoundingRect() const{
		      return m_worldPosition;
	 }

	 const std::vector<QVertex2D> & glCoords()const {return m_tileGLcoords;}
private:
	 std::vector<QVertex2D> m_tileGLcoords;
	 QRect m_imagePosition;
	 QRectF m_worldPosition;
};

#endif
