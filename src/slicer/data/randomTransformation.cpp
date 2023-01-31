#include "randomTransformation.h"
#include <cmath>
#include <QVector2D>
#include "geometry2dtoolbox.h"

RandomTransformation::RandomTransformation(int width, int height,const QPolygonF& poly,const QPolygon& discretPoly,
				const Affine2DTransformation& affine, QObject * parent):QObject(parent),m_affine(affine)
{
	m_width = width;
	m_height=height;
	m_poly= poly;
	m_discretPoly= discretPoly;

}

RandomTransformation::RandomTransformation( int height,const QPolygonF& poly,
				const Affine2DTransformation& affine, QObject * parent):QObject(parent),m_affine(affine)
{
	//m_width = width;
	m_height=height;
	m_poly= poly;


	computeDiscretPoly();
}

		//Copy constructor
RandomTransformation::RandomTransformation(const  RandomTransformation &  r):m_affine(r.m_affine)
{
	this->m_width = r.m_width;
	this->m_height= r.m_height;
	this->m_poly= r.m_poly;
	this->m_discretPoly= r.m_discretPoly;

}
RandomTransformation& RandomTransformation::operator=(const RandomTransformation& r)
{
	this->m_width = r.m_width;
	this->m_height= r.m_height;
	this->m_poly= r.m_poly;
	this->m_discretPoly= r.m_discretPoly;
	this->m_affine = r.m_affine;
	return *this;
}

RandomTransformation::~RandomTransformation()
{

}

void RandomTransformation::computeDiscretPoly()
{
	long nbXLine = m_affine.width();
		long nbInline = m_affine.height();

		QPolygon discreateNodes;
		m_discretPoly.clear(); // just to be safe
		//worldDiscreatePolyLine.clear(); // just to be safe
		for (std::size_t idx=0; idx<m_poly.size(); idx++) {
			double imageI, imageJ;
			m_affine.worldToImage(m_poly[idx].x(), m_poly[idx].y(),imageI, imageJ);
			imageI = std::round(imageI);
			imageJ = std::round(imageJ);
			discreateNodes << QPoint(imageI, imageJ);
		}


		if(discreateNodes.size() != 0){
			for (std::size_t idx=0; idx<discreateNodes.size()-1; idx++) {
				QPointF A = discreateNodes[idx];
				QPointF B = discreateNodes[idx+1];
				long dx = std::abs(A.x() - B.x());
				long dy = std::abs(A.y() - B.y());
				long dirX, dirY;
				dirX = (A.x()<B.x()) ? 1 : -1;
				dirY = (A.y()<B.y()) ? 1 : -1;

				if (dx>dy) {
					for (long i=0; i<=dx;i++) {
						QPoint newPt;
						newPt.setX(A.x()+dirX*i);
						long addY = std::round(((double)(i*dy)) / dx);
						newPt.setY(A.y()+dirY*addY);
						if (newPt.x()>=0 && newPt.x()<nbXLine && newPt.y()>=0 && newPt.y()<nbInline) {
							m_discretPoly << newPt;
							//double newPtXDouble, newPtYDouble;
							//m_affine->imageToWorld(newPt.x(),newPt.y(), newPtXDouble, newPtYDouble);
							//worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
						}
					}
				} else {
					for (long i=0; i<=dy;i++) {
						QPoint newPt;
						newPt.setY(A.y()+dirY*i);
						long addX = std::round(((double)(i*dx)) / dy);
						newPt.setX(A.x()+dirX*addX);
						if (newPt.x()>=0 && newPt.x()<nbXLine && newPt.y()>=0 && newPt.y()<nbInline) {
							m_discretPoly << newPt;
							//double newPtXDouble, newPtYDouble;
							//m_affine->imageToWorld(newPt.x(), newPt.y(), newPtXDouble, newPtYDouble);
							//worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
						}
					}
				}
			}
		}
		m_width = m_discretPoly.size();
}

int RandomTransformation::width()const
{
	return m_width;
}

int RandomTransformation::height()const
{
	return m_height;
}

QPolygonF RandomTransformation::getPoly() const
{
	return m_poly;
}

Affine2DTransformation RandomTransformation::getAffineTransformation() const
{
	return m_affine;
}


QPointF RandomTransformation::worldToImage(QVector3D pos) const
{
	 double worldX =pos.x();// event.worldX();
	    double worldY = pos.z();//event.worldY();


	    double imageI, imageJ;


	    std::size_t indexNearestPoint;
	    double distance = std::numeric_limits<double>::max();

	    m_affine.worldToImage(worldX, worldY, imageI,
	            imageJ); // there may be an issue with the conversion !!!

	    QVector2D ijPt(imageI, imageJ);

	    for (int i=0; i<m_discretPoly.size(); i++) {
	        QVector2D pt(m_discretPoly[i]);
	        if (pt.distanceToPoint(ijPt)<distance) {
	            distance = pt.distanceToPoint(ijPt);
	           // qDebug()<<i<<" distance proj :"<<distance;
	            indexNearestPoint = i;
	        }
	    }

	    std::pair<QPointF, QPointF> line;
	    int indice=0;
	    for (std::size_t idx=m_poly.size()-2; idx<m_poly.size(); idx++)
	    {
	    	double imageI, imageJ;
	    	m_affine.worldToImage(m_poly[idx].x(), m_poly[idx].y(),imageI, imageJ);
	    	//std::get<idx>(line) = QPointF(imageI,imageJ);
	    	if(indice ==0)line.first = QPointF(imageI,imageJ);
	    	else line.second = QPointF(imageI,imageJ);
	    	indice++;

	    }
	    bool ok;
	    QPointF pointIj(ijPt.x(),ijPt.y());
	    std::pair<double, QPointF> proj = getPointProjectionOnLine(pointIj,line,&ok);

	   // qDebug()<<ok<<" proj.first :"<<proj.first<<" , distance :"<<distance;
	    if(proj.first< distance)
	    {
	    	int x1 = std::round(proj.second.x());
	    	int y1 = std::round(proj.second.y());


	    	QVector2D pts(x1,y1);
	    	if(pts.distanceToPoint(ijPt) < 0.01f )
	    	{
	    		distance = proj.first;
	    		QPointF A(pts.x(),pts.y());
				QPointF B(std::round(line.second.x()),std::round(line.second.y()));
				QPointF C(std::round(line.first.x()),std::round(line.first.y()));

				long dx = std::abs(A.x() - B.x());
				long dy = std::abs(A.y() - B.y());
				long dirX, dirY;
				dirX = (A.x()<B.x()) ? -1 : 1;
				dirY = (A.y()<B.y()) ? -1 : 1;

				int dirX2 =(B.x()<C.x()) ? -1 : 1;
				int dirY2 =(B.y()<C.y()) ? -1 : 1;

				if(dirX2 == dirX && dirY2 == dirY)
				{
					//qDebug()<<" distancepoint"<<dx<<" ," <<dy;
					int depas=  std::max(dx,dy);
					indexNearestPoint = m_discretPoly.size()+ depas-1;
				}
	    	}
	    	else
	    	{
	    		QPointF A = proj.second;
				QPointF B(std::round(line.second.x()),std::round(line.second.y()));
				QPointF C(std::round(line.first.x()),std::round(line.first.y()));
				float dx = std::abs(A.x() - B.x());
				float dy = std::abs(A.y() - B.y());

				long dirX, dirY;
				dirX = (A.x()<B.x()) ? -1 : 1;
				dirY = (A.y()<B.y()) ? -1 : 1;

				int dirX2 =(B.x()<C.x()) ? -1 : 1;
				int dirY2 =(B.y()<C.y()) ? -1 : 1;

				if(dirX2 == dirX && dirY2 == dirY)
				{
					QVector2D pt1(floor(dx)*dirX+B.x(), floor(dy)*dirY+B.y());
					QVector2D pt2(ceil(dx)*dirX+B.x(), ceil(dy)*dirY+B.y());

					float distance1 = pt1.distanceToPoint(ijPt);
					float distance2 = pt2.distanceToPoint(ijPt);

					bool mindist1 = true;
					float dist;

					if(distance1 <distance2)
					{
						dist= distance1;
						mindist1 = true;
					}
					else
					{
						dist= distance2;
						mindist1 = false;
					}


					if(dist < distance)
					{
						if(mindist1)
						{
							//qDebug()<<" mindist1"<<dx<<" ," <<dy;
							int depas=  std::max(floor(dx),floor(dy));
							indexNearestPoint = m_discretPoly.size()+ depas-1;
						}
						else
						{
							//qDebug()<<" mindist2"<<dx<<" ," <<dy;
							int depas=  std::max(ceil(dx),ceil(dy));
							indexNearestPoint = m_discretPoly.size()+ depas-1;
						}
					}

				}

	    	}
	    }

	    long newPos = indexNearestPoint;
	    return QPointF(newPos, pos.y());

}

QVector3D RandomTransformation::imageToWorld(QPointF posi)const
{
	double worldX = posi.x();
		double worldY = posi.y();

		double realX, realY;
		QPointF ijPt;
		worldX = std::round(worldX); // round to get the point


		if (worldX>=0 && worldX<m_discretPoly.size()) {
			long idx = worldX;
			ijPt = m_discretPoly[idx];
		} else {
			int depassement = worldX - m_discretPoly.size()+1;

			if(m_poly.size()<2) return QVector3D(0,0,0);

			QPolygon discreateNodes;

			for (std::size_t idx=m_poly.size()-2; idx<m_poly.size(); idx++) {
				double imageI, imageJ;
				m_affine.worldToImage(m_poly[idx].x(), m_poly[idx].y(),imageI, imageJ);
				imageI = std::round(imageI);
				imageJ = std::round(imageJ);
				discreateNodes << QPoint(imageI, imageJ);
			}

			QPointF A = discreateNodes[0];
			QPointF B = discreateNodes[1];
			long dx = std::abs(A.x() - B.x());
			long dy = std::abs(A.y() - B.y());
			long dirX, dirY;
			dirX = (A.x()<B.x()) ? 1 : -1;
			dirY = (A.y()<B.y()) ? 1 : -1;

			if (dx>dy) {
			//	for (long i=0; i<=dx;i++) {
					QPoint newPt;
					newPt.setX(A.x()+dirX*(dx+depassement));
					long addY = std::round(((double)((dx+depassement)*dy)) / dx);
					newPt.setY(A.y()+dirY*addY);
					ijPt = newPt;
					//if (newPt.x()>=0 && newPt.x()<nbXLine && newPt.y()>=0 && newPt.y()<nbInline) {
						//m_discreatePolyLine << newPt;
						//double newPtXDouble, newPtYDouble;
						//dataset->ijToXYTransfo()->imageToWorld(newPt.x(),newPt.y(), newPtXDouble, newPtYDouble);
						//m_worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
					//}
			//	}
			} else {
				//for (long i=0; i<=dy;i++) {
					QPoint newPt;
					newPt.setY(A.y()+dirY*(dx+depassement));
					long addX = std::round(((double)((dx+depassement)*dx)) / dy);
					newPt.setX(A.x()+dirX*addX);
					ijPt = newPt;
					/*if (newPt.x()>=0 && newPt.x()<nbXLine && newPt.y()>=0 && newPt.y()<nbInline) {
						m_discreatePolyLine << newPt;
						double newPtXDouble, newPtYDouble;
						dataset->ijToXYTransfo()->imageToWorld(newPt.x(), newPt.y(), newPtXDouble, newPtYDouble);
						m_worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
					}*/
				//}
			}



		}

		// there may be an issue with the conversion !!!
		m_affine.imageToWorld(ijPt.x(), ijPt.y(), realX, realY);


		return QVector3D(realX, realY, worldY);
}
