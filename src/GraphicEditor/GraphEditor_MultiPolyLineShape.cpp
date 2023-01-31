
#include <QGraphicsSceneMouseEvent>
#include <QPainterPath>
#include <QGraphicsScene>
#include <QDebug>
#include <math.h>
#include <QGraphicsScene>
#include <QGraphicsView>

#include "GraphEditor_GrabberItem.h"
// #include "GraphEditor_PolyLineShape.h"
#include <GraphEditor_MultiPolyLineShape.h>

#include "GraphicSceneEditor.h"
#include "LineLengthText.h"
#include "singlesectionview.h"

#include <malloc.h>

std::vector<std::vector<int>> GraphEditor_MultiPolyLineShape::qPolygonXSplit(QPolygon& polyIn)
{
	std::vector<std::vector<int>> x;
	int N = polyIn.size();
	int n=0;
	bool valid = false;
	std::vector<int> v;
	while ( n < N-1 )
	{
		QPoint p = polyIn.point(n);
		if ( p.y() != GraphEditor_MultiPolyLineShape::NOVALUE )
		{
			if ( !valid )
			{
				valid = true;
				v.clear();
			}
			v.push_back(n);
		}
		else
		{
			if ( valid )
			{
				valid = false;
				if ( !v.empty() ) x.push_back(v);
				v.clear();
			}
		}
		n++;
	}
	if ( !v.empty() ) x.push_back(v);
	return x;
}


std::vector<QPolygonF> GraphEditor_MultiPolyLineShape::qPolygonSplit(QPolygonF& polyIn0)
{
	QPolygon polyIn = polyIn0.toPolygon();
	std::vector<QPolygonF> polygon;
	std::vector<std::vector<int>> X = qPolygonXSplit(polyIn);
	int N = X.size();
	polygon.resize(N);
	for (int n=0; n<N; n++)
	{
		QPolygon p;
		p.resize(X[n].size());
		for (int ix=0; ix<X[n].size(); ix++)
		{
			p.setPoint(ix, polyIn.point(X[n][ix]));
		}
		polygon[n] = p;
	}
	return polygon;
}



GraphEditor_MultiPolyLineShape::GraphEditor_MultiPolyLineShape()
{
	m_textItem = new LineLengthText(this);
	m_textItem->setVisible(false);
	setAcceptHoverEvents(true);
	m_View = nullptr;
	m_IsClosedCurved = false;
	setFlags(ItemIsSelectable|ItemIsFocusable);
	// setFlag(QGraphicsItem::ItemIsSelectable, false);
	m_Menu = nullptr;
	setData(0, "noRotation");
}

void GraphEditor_MultiPolyLineShape::wheelEvent(QGraphicsSceneWheelEvent *event)
{
	//qDebug() << "wheel";
}


/*
GraphEditor_MultiPolyLineShape::GraphEditor_MultiPolyLineShape(QPolygonF polygon, QPen pen, QBrush brush, QMenu* itemMenu,
		GraphicSceneEditor* scene, bool isClosedCurved)
{
	m_scene = scene;
	m_View = nullptr;
	m_IsClosedCurved = isClosedCurved;
	setAcceptHoverEvents(true);
	setFlags(ItemIsSelectable|ItemIsMovable|ItemSendsGeometryChanges|ItemIsFocusable);
	m_Menu = itemMenu;
	setPen(pen);
	m_Pen=pen;
	setBrush(brush);
	m_textItem = new LineLengthText(this);
	setPolygon(polygon);
	m_textItem->setVisible(false);
}
*/

GraphEditor_MultiPolyLineShape::~GraphEditor_MultiPolyLineShape() {
	if(m_nameId !="") emit BezierDeleted(m_nameId);
}

bool GraphEditor_MultiPolyLineShape::checkClosedPath()
{
	if (grabberList.size()>2)
	{
		if (grabberList[grabberList.size()-1]->hasDetectedCollision())
		{
			return true;
		}
	}
	return false;
}

QVector<QPointF> GraphEditor_MultiPolyLineShape::getKeyPoints()
{
	return mapToScene(m_polygon);
}

void GraphEditor_MultiPolyLineShape::insertNewPoints(QPointF pos)
{

	if (m_DrawFinished)
	{
		QPointF clickPos = pos;
		QPolygonF polygonPath = m_polygon;
		QPolygonF newPath( polygonPath );

		bool found = false;
		double distanceMin = 100000;
		QPointF newPointBest;
		double factor =1;
		float epsilon = 0.1f;
		int currentindex = -1;
		int indexAdded;
		for(int i = 0; i < polygonPath.size()-1; i++){ // i go from 0 to N-1 because we do not want to pick on the last line of the polygon
			QPointF p1 = polygonPath.at(i);
			QPointF p2 = (i < polygonPath.size()-1) ? polygonPath.at(i+1) : polygonPath.at(0);
			double APx = clickPos.x() - p1.x();
			double APy = clickPos.y() - p1.y();
			double ABx = p2.x() - p1.x();
			double ABy = p2.y() - p1.y();
			double magAB2 = ABx*ABx + ABy*ABy;
			double ABdotAP = ABx*APx + ABy*APy;
			double t = ABdotAP / magAB2;

			//qDebug()<<" T : "<<t;
			if(t > -epsilon && t < epsilon)
			{
				//qDebug()<<" insertNewPoints  signalCurrentIndex"<<i;
				emit signalCurrentIndex(i);
				return;
			}
			if(t > 1.0-epsilon && t < 1.0+epsilon)
			{
				//qDebug()<<" insertNewPoints  signalCurrentIndex  1+"<<i;
				emit signalCurrentIndex(i+1);
				return;
			}


			QPointF newPoint;

			if ( t < 0) {
				//newPoint = trackLine.p1();
			}else if (t > 1){
				//newPoint = trackLine.p2();
			}else{
				newPoint.setX(p1.x() + ABx*t);
				newPoint.setY(p1.y() + ABy*t);
				double d = sqrt( pow( (newPoint.x() - clickPos.x()), 2) + pow( (newPoint.y() - clickPos.y()), 2));
				if ( d < distanceMin) {
					distanceMin = d;
					newPointBest = newPoint;
					found = true;
					indexAdded = i + 1;
					factor = t;
				}
			}
		}
		if (found) {
			if(currentindex<0) currentindex = indexAdded;
			QPointF nextPoint = (indexAdded - 1 < polygonPath.size()-1) ? polygonPath.at(indexAdded) : polygonPath.at(0);
			int x = polygonPath.at(indexAdded - 1).x() + factor * (nextPoint.x() - polygonPath.at(indexAdded - 1).x());
			int y = polygonPath.at(indexAdded - 1).y() + factor * (nextPoint.y() - polygonPath.at(indexAdded - 1).y());
			QPointF newPoint(x, y );
			newPath.insert(indexAdded, newPoint);
			setPolygon(newPath);
			setGrabbersVisibility(true);
			m_ItemGeometryChanged=true;
		}
	//	qDebug()<<found<<" current index  = "<<currentindex;
		emit signalCurrentIndex(currentindex);
	}


}


void GraphEditor_MultiPolyLineShape::polygonResize(int widthO, int width)
{

	float decal = (width- widthO)/2;

	QPolygonF polygon = m_polygon;
	for(int i=0;i<polygon.size();i++)
	{
		polygon[i].setX(decal+ polygon[i].x());
	}


	for(int i=0;i<grabberList.count();i++)
	{
		grabberList[i]->moveX(decal );
	}

	setPolygonFromMove(polygon);

	//emit polygonChanged(m_polygon);
}

/*
void GraphEditor_MultiPolyLineShape::setPolygon(const std::vector<QPolygonF> &poly)
{

	int N = poly.size();
	m_painterPath.resize(N);
	for (int i=0; i<N; i++)
	{
		QPainterPath newPath;
		newPath.addPolygon(poly[i]);
		if (m_IsClosedCurved)
		{
			newPath.closeSubpath();
		}
		// setPath(newPath);
		// clearGrabbers();
		// showGrabbers();
		// calculatePerimeter();
		if ( i == 0 ) setPolygon(poly[i]);
		m_painterPath[i] = newPath;
		QVector<QPointF> pol = poly[i];
		emit polygonChanged(pol,!m_IsClosedCurved);
	}

}
*/

void GraphEditor_MultiPolyLineShape::setPolygon(const QPolygonF &polygon) {
	if (polygon.isEmpty()) return;

	m_polygon = polygon;
	m_polygonShape = polygonShapeCreate(m_polygon);

	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	if (m_IsClosedCurved)
	{
		newPath.closeSubpath();
	}
	setPath(newPath);
	clearGrabbers();
	showGrabbers();
	calculatePerimeter();

	QVector<QPointF> pol = this->polygon();

	emit polygonChanged(pol,!m_IsClosedCurved);
}

void GraphEditor_MultiPolyLineShape::setPolygonFromMove(const QPolygonF &polygon) {
	m_polygon = polygon;
	QPainterPath newPath;
	newPath.addPolygon(m_polygon);
	if (m_IsClosedCurved)
	{
		newPath.closeSubpath();
	}
	setPath(newPath);
	calculatePerimeter();

	QVector<QPointF> pol =this->polygon();// this->SceneCordinatesPoints();


//	emit polygonChanged(pol,!m_IsClosedCurved);
}

void GraphEditor_MultiPolyLineShape::setDrawFinished(bool value)
{
	m_DrawFinished = value;
	if (grabberList.size()>1)
	{
		if (grabberList[grabberList.size()-1]->hasDetectedCollision())
		{
			m_polygon.removeAt(grabberList.size()-1);
			m_IsClosedCurved=true;
			setPolygon(m_polygon);
		}
	}

}

void GraphEditor_MultiPolyLineShape::polygonChanged1()
{
	//qDebug()<<"GraphEditor_MultiPolyLineShape::polygonChanged1 ";
	emit polygonChanged(m_polygon,!m_IsClosedCurved);
}

void GraphEditor_MultiPolyLineShape::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget){

	painter->setRenderHint(QPainter::Antialiasing,true);
//	painter->setRenderHint(QPainter::HighQualityAntialiasing, true);
	painter->setRenderHint(QPainter::SmoothPixmapTransform, true);

	QBrush brsh = brush();
	if ( !scene() ) return;
	if ( scene()->views().size() == 0 ) return;

	brsh.setTransform(QTransform((scene()->views())[0]->transform().inverted()));
	this->setBrush(brsh);

	if ((option->state & QStyle::State_Selected) || m_IsHighlighted)
	{
		QPen newPen = pen();
		newPen.setWidth(newPen.width()+3);
		painter->setPen(newPen);
		//qDebug() << "paint";
	//	if ( m_IsHighlighted ) qDebug() << "ishighted true"; else qDebug() << "ishighted false";
	}
	else
	{
		painter->setPen(pen());
	}

	if (m_IsClosedCurved)
	{
		painter->setBrush(brush());
	}
	else
	{
		painter->setBrush(QBrush(Qt::NoBrush));
	}

	std::vector<QPolygonF> pf = qPolygonSplit(m_polygon);
	for (int i=0; i<pf.size(); i++)
	{
		QPainterPath newPath;
		newPath.addPolygon(pf[i]);
		painter->drawPath(newPath);
	}
}

void GraphEditor_MultiPolyLineShape::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event) {
	if (m_DrawFinished)
	{
		QPointF clickPos = event->pos();
		QPolygonF polygonPath = m_polygon;
		QPolygonF newPath( polygonPath );

		bool found = false;
		double distanceMin = 10000;
		QPointF newPointBest;
		double factor =1;
		int indexAdded;
		for(int i = 0; i < polygonPath.size()-1; i++){ // i go from 0 to N-1 because we do not want to pick on the last line of the polygon
			QPointF p1 = polygonPath.at(i);
			QPointF p2 = (i < polygonPath.size()-1) ? polygonPath.at(i+1) : polygonPath.at(0);
			double APx = clickPos.x() - p1.x();
			double APy = clickPos.y() - p1.y();
			double ABx = p2.x() - p1.x();
			double ABy = p2.y() - p1.y();
			double magAB2 = ABx*ABx + ABy*ABy;
			double ABdotAP = ABx*APx + ABy*APy;
			double t = ABdotAP / magAB2;

			QPointF newPoint;

			if ( t < 0) {
				//newPoint = trackLine.p1();
			}else if (t > 1){
				//newPoint = trackLine.p2();
			}else{
				newPoint.setX(p1.x() + ABx*t);
				newPoint.setY(p1.y() + ABy*t);
				double d = sqrt( pow( (newPoint.x() - clickPos.x()), 2) + pow( (newPoint.y() - clickPos.y()), 2));
				if ( d < distanceMin) {
					distanceMin = d;
					newPointBest = newPoint;
					found = true;
					indexAdded = i + 1;
					factor = t;
				}
			}
		}
		if (found) {
			QPointF nextPoint = (indexAdded - 1 < polygonPath.size()-1) ? polygonPath.at(indexAdded) : polygonPath.at(0);
			int x = polygonPath.at(indexAdded - 1).x() + factor * (nextPoint.x() - polygonPath.at(indexAdded - 1).x());
			int y = polygonPath.at(indexAdded - 1).y() + factor * (nextPoint.y() - polygonPath.at(indexAdded - 1).y());
			QPointF newPoint(x, y );
			newPath.insert(indexAdded, newPoint);
			setPolygon(newPath);
			setGrabbersVisibility(true);
			m_ItemGeometryChanged=true;
		}
	}
	QGraphicsItem::mouseDoubleClickEvent(event);
}

void GraphEditor_MultiPolyLineShape::showGrabbers() {
	if ( m_readOnly ) return;
	QPolygonF polygonPath = m_polygon;
	for(int i = 0; i < polygonPath.size(); i++){
		QPointF point = polygonPath.at(i);

		GraphEditor_GrabberItem *dot = new GraphEditor_GrabberItem(this);

		connect(dot, &GraphEditor_GrabberItem::signalMove, this, &GraphEditor_MultiPolyLineShape::grabberMove);
		connect(dot, &GraphEditor_GrabberItem::signalDoubleClick, this, &GraphEditor_MultiPolyLineShape::slotDeleted);
		connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_MultiPolyLineShape::polygonChanged1);

		dot->setVisible(true);
		grabberList.append(dot);
		dot->setPos(point);
		if (i==0)
		{
			dot->setFirstGrabberInList(true);
		}
		else if (i == polygonPath.size() -1)
		{
			if (m_IsClosedCurved)
				dot->setDetectCollision(false);
			else
			{
				dot->setDetectCollision(true);
				connect(dot, &GraphEditor_GrabberItem::signalRelease, this, &GraphEditor_MultiPolyLineShape::GrabberMouseReleased);
			}
		}
	}

}

void GraphEditor_MultiPolyLineShape::GrabberMouseReleased(QGraphicsItem *signalOwner)
{
	if ( m_readOnly ) return;
	setDrawFinished(true);
}

void GraphEditor_MultiPolyLineShape::calculatePerimeter()
{
	qreal TotalLength =0;
	if (m_polygon.size()>1)
	{
		for(int i = 1; i < m_polygon.size(); i++)
		{
			QLineF line(m_polygon.at(i-1),m_polygon.at(i));
			TotalLength +=line.length();
		}
		if (m_IsClosedCurved)
		{
			QLineF line(m_polygon.last(),m_polygon.at(0));
			TotalLength +=line.length();
		}
		int polygon_middle;
		if (m_IsClosedCurved)
		{
			polygon_middle=(m_polygon.size()+1)/2-1;
		}
		else
		{
			polygon_middle=m_polygon.size()/2-1;
		}

		QLineF line(m_polygon.at(polygon_middle),m_polygon.at(polygon_middle+1));
		float lineAngle = line.angle();
		if (m_scene)
		{
			if (m_scene->innerView())
			{
				if ((m_scene->innerView()->viewType() == InlineView) ||
						(m_scene->innerView()->viewType() == XLineView) ||
						(m_scene->innerView()->viewType() == RandomView) )
				{
					lineAngle = 360 - lineAngle;
				}
			}
		}
		QPointF textPos;

		if ( lineAngle > 90 && lineAngle < 260)
		{  // Right to left line
			lineAngle -= 180;
		}
		textPos = line.center();
		if ( m_textItem && m_displayPerimetre )
		{
			m_textItem->setText(QString::number(TotalLength)+ " m");
			m_textItem->setPos(textPos);
			m_textItem->setRotation(lineAngle);
			m_textItem->setVisible(true);
		}
	}
}

void GraphEditor_MultiPolyLineShape::grabberMove(QGraphicsItem *signalOwner, qreal dx, qreal dy) {
	if ( m_readOnly ) return;
	m_ItemGeometryChanged=true;
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = m_polygon;
	for(int i = 0; i < grabberList.size(); i++){
		if(grabberList.at(i) == signalOwner){
			QPointF pathPoint = polygonPath.at(i);
			polygonPath.replace(i, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
			emit currentIndexChanged(i);
		}
	}


	setPolygonFromMove(polygonPath);
}



void GraphEditor_MultiPolyLineShape::moveGrabber(int index, int dx, int dy) {
	if ( m_readOnly ) return;
	m_ItemGeometryChanged=true;
		if ( grabberList.isEmpty() )
			return;
	QPolygonF polygonPath = m_polygon;
	QPointF pathPoint = polygonPath.at(index);
	polygonPath.replace(index, QPointF(pathPoint.x() + dx, pathPoint.y() + dy));
	emit currentIndexChanged(index);



	setPolygonFromMove(polygonPath);
}

void GraphEditor_MultiPolyLineShape::positionGrabber(int index, QPointF pos)
{
	if ( m_readOnly ) return;
	m_ItemGeometryChanged=true;
	if ( grabberList.isEmpty() )
		return;
	QPolygonF polygonPath = m_polygon;

	if(index >= 0 && index<grabberList.size())
	{

		grabberList[index]->setPos(pos);
		QPointF pathPoint = polygonPath.at(index);
		polygonPath.replace(index, pos);
		emit currentIndexChanged(index);

		setPolygonFromMove(polygonPath);
	}
	else
		qDebug()<<"GraphEditor_MultiPolyLineShape::positionGrabber:"<<index;

}



GraphEditor_MultiPolyLineShape* GraphEditor_MultiPolyLineShape::clone()
{
	/*
	GraphEditor_MultiPolyLineShape* cloned = new GraphEditor_MultiPolyLineShape(polygon(), pen(), brush(), m_Menu, m_scene,m_IsClosedCurved);
	cloned->setPos(scenePos());
	cloned->setZValue(zValue());
	cloned->setRotation(rotation());
	cloned->setID(m_LayerID);
	cloned->setGrabbersVisibility(false);
	return cloned;
	*/
	return nullptr;
}

QVector<QPointF>  GraphEditor_MultiPolyLineShape::SceneCordinatesPoints()
{
	return mapToScene(polygon());
}

QVector<QPointF>  GraphEditor_MultiPolyLineShape::ImageCordinatesPoints()
{
	QVector<QPointF> vec;
	QPolygonF polygon_ = mapToScene(polygon());
	GraphicSceneEditor *scene_ = dynamic_cast<GraphicSceneEditor*>(scene());
	if (scene_)
	{
		foreach(QPointF p, polygon_) {
			vec.push_back(scene_->innerView()->ConvertToImage(p));
		}
	}
	return vec;
}

bool GraphEditor_MultiPolyLineShape::eventFilter(QObject* watched, QEvent* ev) {
	return false;
}

bool GraphEditor_MultiPolyLineShape::sceneEvent(QEvent *event)
{
	return GraphEditor_Path::sceneEvent(event);
}

void GraphEditor_MultiPolyLineShape::hoverEnterEvent(QGraphicsSceneHoverEvent *event) {
	//qDebug() << "enter";
	if (scene()->selectedItems().empty())
	{
		m_IsHighlighted = true;
		setSelected(true);
	}
	QGraphicsItem::hoverLeaveEvent(event);
}


void GraphEditor_MultiPolyLineShape::hoverLeaveEvent(QGraphicsSceneHoverEvent *event) {
	//qDebug() << "leave";
	m_IsHighlighted = false;
	if (!m_AlreadySelected)
		setSelected(false);
	QGraphicsItem::hoverLeaveEvent(event);
}

void GraphEditor_MultiPolyLineShape::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
	// QGraphicsItem::mouseMoveEvent(event);
}


int GraphEditor_MultiPolyLineShape::getTabIndex(float x)
{
	for (int i=0; i<m_polygon.size()-1; i++)
	{
		float xp1 = m_polygon[i].x();
		float xp2 = m_polygon[i+1].x();
		if ( x >= xp1 && x <= xp2 )
			return i;
	}
	return -1;
}

bool GraphEditor_MultiPolyLineShape::isSelectable(double worldX,double worldY)
{
	int idx = getTabIndex((float)worldX);
	if ( idx < 0 ) return false;

	float yp = m_polygon[idx].y();
	if ( yp == NOVALUE ) return false;
	if ( worldY <= yp+m_searchArea && worldY >= yp-m_searchArea )
	{
		if (scene()->selectedItems().empty())
		{
			m_IsHighlighted = true;
			setSelected(true);
			qDebug() << "hightlight ok";
		}
		return true;
	}
	else
	{
		m_IsHighlighted = false;
		setSelected(false);
		return false;
	}
	return false;
}

void GraphEditor_MultiPolyLineShape::hoverMoveEvent(QGraphicsSceneHoverEvent *event)
{
	// QGraphicsItem::hoverMoveEvent(event);
}

void GraphEditor_MultiPolyLineShape::inputMethodEvent(QInputMethodEvent *event)
{
	qDebug() << "input methode event";
}


bool GraphEditor_MultiPolyLineShape::setSelect(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys)
{
	// return isSelectable(worldX, worldY);
}

bool GraphEditor_MultiPolyLineShape::isSelected()
{
	return m_IsHighlighted;
}


QPolygonF GraphEditor_MultiPolyLineShape::polygonShapeCreate(QPolygonF poly)
{
	/*
	QPolygonF polygonShape;
	for (int i=0; i<poly.size(); i++)
	{
		QPointF valF;
		valF = poly[i].toPoint();
		if ( valF.y() == NOVALUE ) continue;
		valF.setY(valF.y()-m_searchArea);
		polygonShape << valF.toPoint();
	}

	for (int i=poly.size()-1; i>0; i--)
	{
		QPointF valF;
		valF = poly[i].toPoint();
		if ( valF.y() == NOVALUE ) continue;
		valF.setY(valF.y()+m_searchArea);
		polygonShape << valF.toPoint();
	}
	*/

	QPolygonF polygonShape;
	QPolygonF polygon0 = mapToScene(poly);
	GraphicSceneEditor *scene_ = dynamic_cast<GraphicSceneEditor*>(scene());
	if (scene_)
	{
		foreach(QPointF p, polygon0) {
			polygonShape << scene_->innerView()->ConvertToImage(p);
		}
	}

	for (int i=0; i<polygon0.size(); i++)
	{
		QPointF valF;
		valF = polygon0[i].toPoint();
		valF.setY(valF.y()-m_searchArea);
		polygonShape << valF.toPoint();
	}

	for (int i=polygon0.size()-1; i>=0; i--)
	{
		QPointF valF;
		valF = polygon0[i].toPoint();
		valF.setY(valF.y()+m_searchArea);
		polygonShape << valF.toPoint();
	}
	// path.addPolygon(mapToScene(poly));
	return mapFromScene(polygonShape);
}


QPainterPath GraphEditor_MultiPolyLineShape::shape() const
{
    QPainterPath path;
    /*
    QPolygonF poly;

    for (int i=0; i<m_polygon.size(); i++)
    {
    	QPointF valF;
    	valF = m_polygon[i].toPoint();
    	if ( valF.y() == NOVALUE ) continue;
    	valF.setY(valF.y()-m_searchArea);
		poly << valF.toPoint();
    }

    for (int i=m_polygon.size()-1; i!=0; i--)
    {
    	QPointF valF;
    	valF = m_polygon[i].toPoint();
    	if ( valF.y() == NOVALUE ) continue;
    	valF.setY(valF.y()+m_searchArea);
		poly << valF.toPoint();
    }
    */
    path.addPolygon(m_polygonShape);
    return path;
}
