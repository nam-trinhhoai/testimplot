#include "GeRectanglePicking.h"

#include <QApplication>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsScene>
#include <QKeyEvent>
#include <QDebug>

#include <qgraphicsitem.h>

#include "GeEllipse.h"
#include "GePolygon.h"
#include "GeRectangle.h"

#include "GeObjectId.h"

const GeObjectId GeRectanglePicking::selectionNilObjectId = GeObjectId(0,0);

GeRectanglePicking::GeRectanglePicking(	GeGlobalParameters& globalParameters,
		QObject* parent) :
		PickingTask( parent),
		m_currentItem(nullptr),
		m_currentAction(DefaultType),
		m_previousAction(0),
		m_leftMouseButtonPressed(false),
		currentObjectId(selectionNilObjectId),
		//viewer(parent),
		globalParameters( globalParameters)//,
		//vaData (vaData)
{
	//canvas = parent->getScene();
	//m_currentAction = ActionTypes::RectangleType;
	//connect(canvas, &QGraphicsScene::selectionChanged, this, &GeRectanglePicking::checkSelection);
}

GeRectanglePicking::~GeRectanglePicking() {
    //if ( currentItem) delete currentItem;
}


void GeRectanglePicking::checkSelection() {
}

int GeRectanglePicking::currentAction() const {
    return m_currentAction;
}

QPointF GeRectanglePicking::previousPosition() const
{
    return m_previousPosition;
}

void GeRectanglePicking::setCurrentAction(const int type)
{
    m_currentAction = type;
}

void GeRectanglePicking::setCurrentAction2(const int type, int objectIndex, QColor& color)
{
	m_currentColor = color;
    m_currentAction = type;
    int slice = 0; //TODO viewer->getSliceToBeAnalyse();
    //currentObjectId.setSlice(viewer->getSliceToBeAnalyse());
    //currentObjectId.setObjectId(objectIndex);
}

void GeRectanglePicking::setColor(const QColor& c)
{
    m_currentColor = c;
}

void GeRectanglePicking::setPreviousPosition(const QPointF previousPosition)
{
    if (m_previousPosition == previousPosition)
        return;

    m_previousPosition = previousPosition;
    emit previousPositionChanged();
}

void GeRectanglePicking::mousePressed(double worldX,double worldY,
		Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) {

    if (button == Qt::LeftButton) {
        m_leftMouseButtonPressed = true;
        setPreviousPosition(QPointF(worldX, worldY));
        if(QApplication::keyboardModifiers() & Qt::ShiftModifier){
            m_previousAction = m_currentAction;
            QColor c(0,0,0,0);
            setCurrentAction2(SelectionType, -1, c);
        }
        m_indexColor++;
        if(m_indexColor> 3)
            m_indexColor = 0;
    }

	if (m_leftMouseButtonPressed && !(button & Qt::RightButton) && !(button & Qt::MiddleButton)) {
		deselectItems();

		GeRectangle* rectangle = new GeRectangle(
				globalParameters,
			/*vaData->getObjectName(currentObjectId*/"MyRectangle1",
			currentObjectId, m_currentColor, true, this);

		m_currentItem = rectangle;
		m_canvas->addItem(m_currentItem);
		connect(rectangle, &GeRectangle::clicked, this, &GeRectanglePicking::signalSelectItem);
		connect(rectangle, &GeRectangle::signalMove, this, &GeRectanglePicking::slotMove);

		connect(rectangle, &GeRectangle::itemSelected, this, &GeRectanglePicking::objectSelectedFromItem);

		emit signalNewSelectItem(rectangle, m_indexColor);
//		QObject::connect(rectangle, &GeRectangle::rectangleChanged, this, [=](const QRectF& rect,
//				GeObjectId& modifiedObjectId, bool b) {
//			emit GeRectanglePicking::rectangleChanged(rect, modifiedObjectId.getObjectId(), this);
//		});

		rectangle->setSelected(true);
	}
}

void GeRectanglePicking::mouseMoved(
		double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,
		const QVector<PickingInfo> & info) {

	if (m_leftMouseButtonPressed) {
		auto dx = worldX - m_previousPosition.x();
		auto dy = worldY - m_previousPosition.y();
		GeRectangle * rectangle = qgraphicsitem_cast<GeRectangle *>(m_currentItem);
		rectangle->setRect((dx > 0) ? m_previousPosition.x() : worldX,
						   (dy > 0) ? m_previousPosition.y() : worldY,
						   qAbs(dx), qAbs(dy));
	}
}

void GeRectanglePicking::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
//    if (button & Qt::LeftButton) m_leftMouseButtonPressed = false;
//
//	if (!m_leftMouseButtonPressed&&
//			!(button & Qt::RightButton) &&
//			!(button & Qt::MiddleButton)) {
//		GeRectangle * rectangle = qgraphicsitem_cast<GeRectangle *>(m_currentItem);
//		rectangle->setPositionGrabbers();
//		emit signalSelectItem(rectangle);
//		// Rectangle creation in data
//		emit GeRectanglePicking::rectangleChanged(rectangle->rect(),
//				rectangle->getObjectId().getObjectId(), this);
//
//	}
//	QColor c(0,0,0,0);
//	setCurrentAction2(GeRectanglePicking::DefaultType, -1, c);
}

void GeRectanglePicking::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{

}

void GeRectanglePicking::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
    case Qt::Key_Delete: {
        foreach (QGraphicsItem *item, m_canvas->selectedItems()) {
  //          signalRemoveItem(item);
 //           canvas->removeItem(item);

            //TODO question HASSAN    QObject::disconnect(item, &GeRectangle::rectangleChanged, this);
            //emit itemDeletedSignal(item);
     //       delete item;
        }
        deselectItems();
        break;
    }
    case Qt::Key_A: {
        if(QApplication::keyboardModifiers() & Qt::ControlModifier){
            foreach (QGraphicsItem *item, m_canvas->items()) {
                item->setSelected(true);
            }
            if(m_canvas->selectedItems().length() == 1) signalSelectItem(m_canvas->selectedItems().at(0));
        }
        break;
    }
    default:
        break;
    }
    //SceneExtension::keyPressEvent(event);
}

void GeRectanglePicking::objectSelectedFromItem(int ojectId) {
	emit GeRectanglePicking::objectSelectedSignal(ojectId);
}

void GeRectanglePicking::deselectItems() {
    foreach (QGraphicsItem *item, m_canvas->selectedItems()) {
        item->setSelected(false);
    }
    m_canvas->selectedItems().clear();
}

/**
 * Top Down
 */
void GeRectanglePicking::propagSelectionOnGraphics(int objectId) {

    foreach (QGraphicsItem *item, m_canvas->items()) {
    	GeShape* shape = dynamic_cast<GeShape*>(item);
    	if (shape) {
    		if ( shape->getObjectId().getObjectId() == objectId ) {
    			item->setSelected(true);
    		} else {
    			if (item->isSelected()) {
    				item->setSelected(false);
    			}
    		}
    	}
    }
}

void GeRectanglePicking::slotMove(QGraphicsItem *signalOwner, qreal dx, qreal dy)
{
//    foreach (QGraphicsItem *item, canvas->selectedItems()) {
//        if(item != signalOwner) item->moveBy(dx,dy);
//    }
}

void GeRectanglePicking::cleanScene() {
	foreach (QGraphicsItem *item, m_canvas->items()) {
		GeShape *node = dynamic_cast<GeShape *>(item);
	    if (node)
	    	m_canvas->removeItem( item );
	}
}

void GeRectanglePicking::cleanObjectGraphicsOnSlice(int objectId) {
	bool found = false;

	foreach (QGraphicsItem *item, m_canvas->items()) {
		GeShape* shape = dynamic_cast<GeShape*>(item);
		if (shape) {
			if ( shape->getObjectId().getObjectId() == objectId ) {
				m_canvas->removeItem( item );
				break;
			}
		 }
	}
}

/**
 * After slice change
 */
void GeRectanglePicking::drawAllGraphicsItems() {
//	SyncViewer2d* view2d = getViewer();
//	QGraphicsView* graphicsView =  view2d->getScene()->views().first();
//
//	int currentSlice = view2d->getSliceToBeAnalyse();
//	int objectsSize = vaData->getObjectsSize();
//	// Boucle sur les objets
//	for (int currentObjectIndex = 0; currentObjectIndex < objectsSize; currentObjectIndex++) {
//		QString currentObjectName = vaData->getObjectName(currentObjectIndex);
//		int sliceMin, sliceMax;
//		vaData->getObjectSliceMinMax(currentObjectIndex, sliceMin, sliceMax);
//		if ( sliceMin > currentSlice || sliceMax < currentSlice) continue;
//
//		switch ( vaData->getObjectType(currentObjectIndex) ) {
//		case data::MtVaObjectType::Rectangle :
//			{
//				rectDisplay(currentObjectIndex, currentSlice, false);
//			}
//			break;
//		case data::MtVaObjectType::Polygon :
//			{
//				polygonDisplay(currentObjectIndex, currentSlice, false);
//			}
//			break;
//		case data::MtVaObjectType::Ellipse :
//			{
//				ellipseDisplay(currentObjectIndex, currentSlice, false);
//			}
//			break;
//		}
//	}
}

/**
 * After picking or move
 */
void GeRectanglePicking::refreshObjectDisplay(int objectId) {
//	SyncViewer2d* view2d = getViewer();
//	QGraphicsView* graphicsView =  view2d->getScene()->views().first();
//
//	int currentSlice = view2d->getSliceToBeAnalyse();
//	// Boucle sur les objets
//
//	QColor c = vaData->getObjectColor(objectId);
//	QString currentObjectName = vaData->getObjectName(objectId);
//
//	int sliceMin, sliceMax;
//	vaData->getObjectSliceMinMax(objectId, sliceMin, sliceMax);
//	if ( sliceMin > currentSlice || sliceMax < currentSlice) {
//		cleanObjectGraphicsOnSlice(objectId);
//		return;
//	}
//
//	switch ( vaData->getObjectType(objectId) ) {
//	case data::MtVaObjectType::Rectangle : {
//			rectDisplay(objectId, currentSlice, true);
//		}
//		break;
//	case data::MtVaObjectType::Polygon : {
//			polygonDisplay(objectId, currentSlice, true);
//		}
//		break;
//	case data::MtVaObjectType::Ellipse : {
//			ellipseDisplay(objectId, currentSlice, true);
//		}
//		break;
//	}
}

/**
 * Single object
 */
void GeRectanglePicking::rectDisplay(int objectId, int currentSlice, bool removeBefore) {
//	QRectF currentRect;
//	QColor c = vaData->getObjectColor(objectId);
//	if ( ! vaData->findRect(currentSlice, objectId, currentRect)) {
//		// Object does not exists on this slice
//		int beforeIndex, nextIndex;
//		QRectF rectBefore(0,0,0,0), rectNext(0,0,0,0) ;
//		if ( vaData->findRectObjectBeforeAfter(objectId, currentSlice,
//				beforeIndex, rectBefore, nextIndex, rectNext)) {
//			if (nextIndex == data::MtVaObject::UNDEF_AFTER_SLICE && beforeIndex < currentSlice) {
//				refreshOrCreateItem(objectId, rectBefore, c, false, removeBefore );
//			}
//
//			else if (beforeIndex == data::MtVaObject::UNDEF_BEFORE_SLICE  && nextIndex > currentSlice) {
//				refreshOrCreateItem(objectId, rectNext, c, false, removeBefore );
//
//			} else {
//				QPointF pul0 = rectBefore.topLeft();
//				QPointF pul1 = rectNext.topLeft();
//				float xuli = ((currentSlice - beforeIndex) * (pul1.x() - pul0.x()) / (nextIndex - beforeIndex)) + pul0.x();
//				float yuli = ((currentSlice - beforeIndex) * (pul1.y() - pul0.y()) / (nextIndex - beforeIndex)) + pul0.y();
//
//				QPointF plr0 = rectBefore.bottomRight();
//				QPointF plr1 = rectNext.bottomRight();
//				float xlri = ((currentSlice - beforeIndex) * (plr1.x() - plr0.x()) / (nextIndex - beforeIndex)) + plr0.x();
//				float ylri = ((currentSlice - beforeIndex) * (plr1.y() - plr0.y()) / (nextIndex - beforeIndex)) + plr0.y();
//
//				QRectF interpRect(xuli, yuli, (xlri - xuli + 1), (ylri - yuli + 1));
//
//				refreshOrCreateItem(objectId, interpRect, c, false, removeBefore );
//			}
//		}
//		else {
//			//	NON TROUVE
//		}
//	}
//	else { // Il y a un rectangle sur cette slice il faut le dessiner
//		refreshOrCreateItem(objectId, currentRect, c, true, removeBefore );
//	}
}




void GeRectanglePicking::refreshOrCreateItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere, bool removeBefore) {
//	bool found = false;
//
//	if ( removeBefore ) {
//		foreach (QGraphicsItem *item, canvas->items()) {
//			GeShape* shape = dynamic_cast<GeShape*>(item);
//			if (shape) {
//				if ( shape->getObjectId().getObjectId() == objectId ) {
//					canvas->removeItem( item );
//					if ( !found ) createItem(objectId, newRect, color, pickedHere);
//
//					found = true;
//					break;
//				}
//			 }
//		}
//	}
//	if (!found) {
//		// Create
//		createItem( objectId, newRect, color, pickedHere);
//	}
}

void GeRectanglePicking::refreshOrCreateItem(int objectId, QPolygonF& newRect, QColor& color, bool pickedHere, bool removeBefore) {
//	bool found = false;
//
//	if ( removeBefore ) {
//		foreach (QGraphicsItem *item, canvas->items()) {
//			MtGraphicsPolygon* polygon_cast = dynamic_cast<MtGraphicsPolygon*>(item);
//			if (polygon_cast) {
//				if ( polygon_cast->getObjectId().getObjectId() == objectId ) {
//					canvas->removeItem( item );
//					if ( !found ) createItem(objectId, newRect, color, pickedHere);
//
//					found = true;
//					break;
//				}
//			 }
//		}
//	}
//	if (!found) {
//		// Create
//		createItem( objectId, newRect, color, pickedHere);
//	}
}

void GeRectanglePicking::refreshOrCreateEllipseItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere, bool removeBefore) {
//	bool found = false;
//
//	if ( removeBefore ) {
//		foreach (QGraphicsItem *item, canvas->items()) {
//			GeEllipse* ellipseCast = dynamic_cast<GeEllipse*>(item);
//			if (ellipseCast) {
//				if ( ellipseCast->getObjectId().getObjectId() == objectId ) {
//					canvas->removeItem( item );
//					if ( !found ) createEllipseItem(objectId, newRect, color, pickedHere);
//
//					found = true;
//					break;
//				}
//			 }
//		}
//	}
//	if (!found) {
//		// Create
//		createEllipseItem( objectId, newRect, color, pickedHere);
//	}
}

void GeRectanglePicking::createItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere) {
//	VaObjectId oId(objectId, 23);
//	GeRectangle *rectangle = new GeRectangle(vaData, globalParameters, vaData->getObjectName(objectId),
//			oId, color, pickedHere, newRect, this);
//	m_currentItem = rectangle;
//	canvas->addItem(m_currentItem);
//	connect(rectangle, &GeRectangle::clicked, this, &GeRectanglePicking::signalSelectItem);
//	connect(rectangle, &GeRectangle::signalMove, this, &GeRectanglePicking::slotMove);
//
//	connect(rectangle, &GeRectangle::itemSelected, this, &GeRectanglePicking::objectSelectedFromItem);
//
//	// emit signalNewSelectItem(rectangle, m_indexColor);
//	QObject::connect(rectangle, &GeRectangle::rectangleChanged, this, [=](const QRectF& rect,
//			VaObjectId& modifiedObjectId, bool b) {
//		emit GeRectanglePicking::rectangleChanged(rect, modifiedObjectId.getObjectId(), this);
//	});
//	QColor c(0,0,0,0);
//	setCurrentAction2(GeRectanglePicking::DefaultType, -1, c);
}

void GeRectanglePicking::createItem(int objectId, QPolygonF& newPolygon, QColor& color, bool pickedHere) {
//	VaObjectId oId(objectId, 23);
//	MtGraphicsPolygon *polygon = new MtGraphicsPolygon(vaData, globalParameters, oId, color, pickedHere, this);
//	polygon->setPolygon(newPolygon);
//	m_currentItem = polygon;
//	canvas->addItem(m_currentItem);
//	connect(polygon, &MtGraphicsPolygon::clicked, this, &GeRectanglePicking::signalSelectItem);
//	connect(polygon, &MtGraphicsPolygon::signalMove, this, &GeRectanglePicking::slotMove);
//	connect(polygon, &MtGraphicsPolygon::polygonPointAddedSignal, this, &GeRectanglePicking::polygonPointAdded);
//	connect(polygon, &MtGraphicsPolygon::polygonPointDeletedSignal, this, &GeRectanglePicking::polygonPointDeleted);
//	connect(polygon, &MtGraphicsPolygon::itemSelected, this, &GeRectanglePicking::objectSelectedFromItem);
//
//	// emit signalNewSelectItem(rectangle, m_indexColor);
//	QObject::connect(polygon, &MtGraphicsPolygon::polygonChanged, this, [=](const QPolygonF& poly,
//			VaObjectId& modifiedObjectId, bool b, MtGraphicsPolygon* originGraphicsItem) {
//		emit GeRectanglePicking::polygonChanged(poly, modifiedObjectId.getObjectId(), this);
//	});
//	QColor c(0,0,0,0);
//	setCurrentAction2(GeRectanglePicking::DefaultType, -1, c);
}


void GeRectanglePicking::setSlice(int slice) {
	    qDebug() << "SET SLICE" << slice;
}

void GeRectanglePicking::initCanvas(QGraphicsScene* canvas) {
	//canvas->addItem(this);
	this->m_canvas = canvas;
	//canvas->addItem(&editableRect);
}

void GeRectanglePicking::releaseCanvas(QGraphicsScene* canvas) {
//	canvas->removeItem(this);
	//canvas->removeItem(&editableRect);
}

