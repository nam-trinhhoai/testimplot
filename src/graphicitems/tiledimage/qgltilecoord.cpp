#include "qgltilecoord.h"

QGLTileCoord::QGLTileCoord()
{
}

QGLTileCoord::QGLTileCoord(const QRect &imagePosition, const QRectF & worldPosition,const  std::vector<QVertex2D> &glCoords) {
    this->m_imagePosition = imagePosition;
    this->m_worldPosition = worldPosition;
    m_tileGLcoords.clear();
	m_tileGLcoords.resize(glCoords.size());
	for(int i=0;i<m_tileGLcoords.size();i++)
	{
		const QVertex2D  v =glCoords[i];
		m_tileGLcoords[i]=QVertex2D(v.position,v.coords);
	}
}

QGLTileCoord & QGLTileCoord::operator= ( const QGLTileCoord & val )
{
	if (this != &val) {
		this->m_imagePosition=val.m_imagePosition;
		this->m_worldPosition=val.m_worldPosition;

		m_tileGLcoords.clear();
		m_tileGLcoords.resize(val.m_tileGLcoords.size());
		for(int i=0;i<m_tileGLcoords.size();i++)
		{
			const QVertex2D  v =val.m_tileGLcoords[i];
			m_tileGLcoords[i]=QVertex2D(v.position,v.coords);
		}
	}
	return *this;
}

bool QGLTileCoord::operator==(const QGLTileCoord  & rhs) const
{
	return this->m_imagePosition==rhs.m_imagePosition;
}

QGLTileCoord::QGLTileCoord(const QGLTileCoord &tc) {
	m_imagePosition=tc.imageBoundingRect();
	m_worldPosition=tc.worldBoundingRect();

	m_tileGLcoords.clear();
	m_tileGLcoords.resize(tc.m_tileGLcoords.size());
	for(int i=0;i<m_tileGLcoords.size();i++)
	{
		const QVertex2D  v =tc.m_tileGLcoords[i];
		m_tileGLcoords[i]=QVertex2D(v.position,v.coords);
	}
 }

QGLTileCoord::~QGLTileCoord() {

}



