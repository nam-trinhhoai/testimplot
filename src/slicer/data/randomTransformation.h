#ifndef RandomTransformation_H
#define RandomTransformation_H

#include <QObject>
#include <QPolygonF>
#include "affine2dtransformation.h"
#include <QVector3D>

class RandomTransformation: public QObject
{
Q_OBJECT



public:

RandomTransformation(int width, int height,const QPolygonF& poly,const QPolygon& discretPoly, const Affine2DTransformation& affine, QObject * parent=nullptr);
RandomTransformation(int height,const QPolygonF& poly, const Affine2DTransformation& affine, QObject * parent=nullptr);

		//Copy constructor
RandomTransformation(const  RandomTransformation & );
RandomTransformation& operator=(
				const RandomTransformation&);

	~RandomTransformation();


	QPointF worldToImage(QVector3D pos) const ;
	QVector3D imageToWorld(QPointF posi)const  ;


	int width()const;
	int height()const;


	QPolygonF getPoly() const;
	Affine2DTransformation getAffineTransformation() const;

	void computeDiscretPoly();
private:
	int m_width;
	int m_height;
	QPolygonF m_poly;
	QPolygon m_discretPoly;
	Affine2DTransformation m_affine;

};


#endif
