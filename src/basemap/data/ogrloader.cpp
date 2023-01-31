#include "ogrloader.h"
#include <iostream>
#include <QGraphicsScene>
#include <QVector2D>

#include "qglpolylineitem.h"
#include "qglpolygonitem.h"
#include "qglpointclouditem.h"

#include "gdal.h"
#include "ogrsf_frmts.h"
#include "ogr_geometry.h"
#include "ogr_feature.h"

OGRLoader::OGRLoader() {
	// TODO Auto-generated constructor stub

}

OGRLoader::~OGRLoader() {
	// TODO Auto-generated destructor stub
}
 void OGRLoader::loadFile(const std::string & path,QList<QGraphicsItem *>& items )
{
	GDALDataset *poDS;
	poDS = (GDALDataset*) GDALOpenEx(path.c_str(),
			GDAL_OF_VECTOR, NULL, NULL, NULL);
	if (poDS == NULL) {
		std::cerr << "Failed to open Vector file: " << path
				<< std::endl;
		return;
	}


	OGRLayer *poLayer = poDS->GetLayer(0);
	OGRFeature *poFeature = poLayer->GetNextFeature();
	while (poFeature != nullptr) {
		OGRGeometry *geom = poFeature->GetGeometryRef();

		OGRwkbGeometryType t=geom->getGeometryType();

		if (t==OGRwkbGeometryType::wkbLineString ||t==OGRwkbGeometryType::wkbLineString25D || t==OGRwkbGeometryType::wkbLineStringM || t==OGRwkbGeometryType::wkbLineStringZM) {
			if(OGRLineString *el = dynamic_cast<OGRLineString*>(geom))addLine(el,items);
		}else if(t==OGRwkbGeometryType::wkbPoint ||t==OGRwkbGeometryType::wkbPoint25D || t==OGRwkbGeometryType::wkbPointM || t==OGRwkbGeometryType::wkbPointZM)
		{
			if(OGRPoint *el = dynamic_cast<OGRPoint*>(geom))addPoint(el,items);
		}else if(t==OGRwkbGeometryType::wkbPolygon || t==OGRwkbGeometryType::wkbPolygon25D || t==OGRwkbGeometryType::wkbPolygonM || t==OGRwkbGeometryType::wkbPolygonZM )
		{
			if(OGRPolygon *el = dynamic_cast<OGRPolygon*>(geom))addPolygon(el,items);
			//return;
		}else if(t==OGRwkbGeometryType::wkbMultiLineString ||t==OGRwkbGeometryType::wkbMultiLineString25D || t==OGRwkbGeometryType::wkbMultiLineStringM || t==OGRwkbGeometryType::wkbMultiLineStringZM)
		{
			if(OGRGeometryCollection *col = dynamic_cast<OGRGeometryCollection*>(geom))
			{
				for(int i=0;i<col->getNumGeometries();i++)
				{
					addLine(dynamic_cast<OGRLineString*>(col->getGeometryRef(i)),items);
				}
			}
		}else if(t==OGRwkbGeometryType::wkbMultiPoint ||t==OGRwkbGeometryType::wkbMultiPoint25D || t==OGRwkbGeometryType::wkbMultiPointM || t==OGRwkbGeometryType::wkbMultiPointZM)
		{
			if(OGRGeometryCollection *col = dynamic_cast<OGRGeometryCollection*>(geom))
			{
				for(int i=0;i<col->getNumGeometries();i++)
				{
					addPoint(dynamic_cast<OGRPoint*>(col->getGeometryRef(i)),items);
				}
			}

		}else if(t==OGRwkbGeometryType::wkbMultiPolygon || t==OGRwkbGeometryType::wkbMultiPolygon25D || t==OGRwkbGeometryType::wkbMultiPolygonM || t==OGRwkbGeometryType::wkbMultiPolygonZM )
		{
			if(OGRGeometryCollection *col = dynamic_cast<OGRGeometryCollection*>(geom))
			{
				for(int i=0;i<col->getNumGeometries();i++)
				{
					addPolygon(dynamic_cast<OGRPolygon*>(col->getGeometryRef(i)),items);
				}
			}

		}else
			std::cout<<"Geometry type not yet handled"<<std::endl;

		poFeature = poLayer->GetNextFeature();
	}

	if (poDS)
		GDALClose(poDS);
}


void OGRLoader::addLine(OGRSimpleCurve * el,QList<QGraphicsItem *>& items)
{
	OGREnvelope ext;
	el->getEnvelope(&ext);

	QVector<QVector2D> points(el->getNumPoints());
	OGRPointIterator *it = el->getPointIterator();
	OGRPoint p;
	int i=0;
	while (it->getNextPoint(&p)) {
		points[i++]=QVector2D(p.getX(), p.getY());
	}

	QGLPolylineItem *item=new QGLPolylineItem(points,QRectF(ext.MinX,ext.MinY,ext.MaxX-ext.MinX,ext.MaxY-ext.MinY));
	items.push_back(item);
}

void OGRLoader::addPoint(OGRPoint * el,QList<QGraphicsItem *>& items)
{
	OGREnvelope ext;
	el->getEnvelope(&ext);

	QVector<QVector2D> points(1);
	points[0]=QVector2D(el->getX(), el->getY());

	QGLPointCloudItem *item=new QGLPointCloudItem(points,QRectF(ext.MinX,ext.MinY,ext.MaxX-ext.MinX,ext.MaxY-ext.MinY));
	items.push_back(item);
}

void OGRLoader::addPolygon(OGRPolygon * poly,QList<QGraphicsItem *>& items)
{
	OGREnvelope ext;
	poly->getEnvelope(&ext);

	OGRLinearRing *el = poly->getExteriorRing();
	QVector<QVector2D> points(el->getNumPoints());
	OGRPointIterator *it = el->getPointIterator();
	OGRPoint p;
	int i=0;
	while (it->getNextPoint(&p)) {
		points[i++]=QVector2D(p.getX(), p.getY());
	}
	QGLPolygonItem *item=new QGLPolygonItem(points,QRectF(ext.MinX,ext.MinY,ext.MaxX-ext.MinX,ext.MaxY-ext.MinY));
	items.push_back(item);
}

