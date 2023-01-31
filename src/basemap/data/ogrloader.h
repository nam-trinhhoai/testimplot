#ifndef RGTSEISMICSLICER_SRC_BASEMAP_OGRLOADER_H_
#define RGTSEISMICSLICER_SRC_BASEMAP_OGRLOADER_H_

#include <string>
#include <QList>

class QGraphicsItem;
class OGRSimpleCurve;
class OGRPoint;
class OGRPolygon;
class OGRLoader {
public:
	virtual ~OGRLoader();
	static void loadFile(const std::string & path,QList<QGraphicsItem *>& items );
private :
	OGRLoader();

private:
	static void addLine(OGRSimpleCurve * curve,QList<QGraphicsItem *>& items);
	static void addPoint(OGRPoint * el,QList<QGraphicsItem *>& items);
	static void addPolygon(OGRPolygon * el,QList<QGraphicsItem *>& items);
};

#endif /* RGTSEISMICSLICER_SRC_BASEMAP_OGRLOADER_H_ */
