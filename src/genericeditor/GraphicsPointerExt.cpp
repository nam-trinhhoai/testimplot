#include "GraphicsPointerExt.h"

#include <QApplication>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QKeyEvent>
#include <QDebug>

#include <qgraphicsitem.h>
#include "../genericeditor/GeEllipse.h"
#include "../genericeditor/GePolygon.h"
#include "../genericeditor/GePolyLine.h"
#include "../genericeditor/GeRectangle.h"
#include "abstract2Dinnerview.h"
//#include "veselectionrect.h"
//#include "vepolyline.h"
//#include "vepen.h"
//#include "viewers/videoeditor/VaObjectId.h"
//#include "viewers/view2d/SyncViewer2d.h"
//#include "data/MtVaData.h"


//const VaObjectId GraphicsPointerExt::selectionNilObjectId = VaObjectId(0,0);

SceneEventBlocker::SceneEventBlocker(QList<QEvent::Type> blockList, QObject* parent) :
		QObject(parent), m_blockTypes(blockList) {
}

SceneEventBlocker::~SceneEventBlocker() {}

bool SceneEventBlocker::eventFilter(QObject* item, QEvent* event) {
	if (m_blockTypes.contains(event->type())) {
		return true;
	}
	return false;
}

//QRectF SceneEventBlocker::boundingRect() const {
//	return QRectF();
//}
//
//void SceneEventBlocker::paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*) {}

GraphicsPointerExt::GraphicsPointerExt( /*data::MtVaData* vaData,*/ GeGlobalParameters& globalParameters, Abstract2DInnerView *parent) :
	PickingTask(parent),
    currentItem(nullptr),
    m_currentAction(DefaultType),
    m_previousAction(0),
    m_leftMouseButtonPressed(false),
//	currentObjectId(selectionNilObjectId),
	viewer(parent),
	globalParameters( globalParameters),//,
	//vaData (vaData)
	m_eventBlocker({QEvent::GraphicsSceneMouseDoubleClick, QEvent::GraphicsSceneMousePress, QEvent::GraphicsSceneMouseRelease})
{
	canvas = parent->scene();
	m_currentAction = ActionTypes::RectangleType;
	m_drawingType = DrawingType::Rectangle;
	connect(canvas, &QGraphicsScene::selectionChanged, this, &GraphicsPointerExt::checkSelection);

	canvas->installEventFilter(&m_eventBlocker);
}

GraphicsPointerExt::~GraphicsPointerExt() {
	canvas->removeEventFilter(&m_eventBlocker);
    if ( currentItem) delete currentItem;
}


void GraphicsPointerExt::checkSelection() {
}

int GraphicsPointerExt::currentAction() const {
    return m_currentAction;
}

QPointF GraphicsPointerExt::previousPosition() const
{
    return m_previousPosition;
}

void GraphicsPointerExt::setCurrentAction(const int type)
{
    m_currentAction = type;
    switch(m_currentAction) {
    case PolygonType:
    	m_drawingType = DrawingType::Polygon;
    	break;
    case PolyLineType:
    	m_drawingType = DrawingType::PolyLine;
    	break;
    case EllipseType:
    	m_drawingType = DrawingType::Ellipse;
    	break;
    case RectangleType:
    	m_drawingType = DrawingType::Rectangle;
    	break;
    default:
    	m_drawingType = DrawingType::NoType;
    }
    currentItem = nullptr; // item should be in scene so no memory leak
}

void GraphicsPointerExt::setCurrentAction2(const int type, int objectIndex, QColor& color)
{
	m_currentColor = color;
    m_currentAction = type;

    switch(m_currentAction) {
    case PolygonType:
    	m_drawingType = DrawingType::Polygon;
    	break;
    case PolyLineType:
    	m_drawingType = DrawingType::PolyLine;
    	break;
    case EllipseType:
    	m_drawingType = DrawingType::Ellipse;
    	break;
    case RectangleType:
    	m_drawingType = DrawingType::Rectangle;
    	break;
    default:
    	m_drawingType = DrawingType::NoType;
    }
    currentItem = nullptr; // item should be in scene so no memory leak
//    int slice = viewer->getSliceToBeAnalyse();
//    currentObjectId.setSlice(viewer->getSliceToBeAnalyse());
//    currentObjectId.setObjectId(objectIndex);
}

void GraphicsPointerExt::setColor(const QColor& c)
{
    m_currentColor = c;
}

void GraphicsPointerExt::setPreviousPosition(const QPointF previousPosition)
{
    if (m_previousPosition == previousPosition)
        return;

    m_previousPosition = previousPosition;
    emit previousPositionChanged();
}

void GraphicsPointerExt::mousePressed(double worldX,double worldY,Qt::MouseButton button,
		Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info) {

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

    switch (m_currentAction) {
    case PolygonType: {
        if (m_leftMouseButtonPressed && !(button == Qt::RightButton) &&
        		!(button == Qt::MiddleButton)) {
        	if (!polygonOn) {
        		// First point
        		deselectItems();
        		GeObjectId objectId(1, 1);
        		GePolygon *polygon = new GePolygon(/*vaData,*/ globalParameters, objectId, m_currentColor, true);
				currentItem = polygon;
				canvas->addItem(currentItem);
				currentItem->setZValue(m_defaultZ);
				connect(polygon, &GePolygon::clicked, this, &GraphicsPointerExt::signalSelectItem);
				connect(polygon, &GePolygon::signalMove, this, &GraphicsPointerExt::slotMove);
				connect(polygon, &GePolygon::polygonPointAddedSignal, this, &GraphicsPointerExt::polygonPointAdded);
				connect(polygon, &GePolygon::polygonPointDeletedSignal, this, &GraphicsPointerExt::polygonPointDeleted);
				connect(polygon, &GePolygon::itemSelected, this, &GraphicsPointerExt::objectSelectedFromItem);

				emit signalNewSelectItem(polygon, m_indexColor);
				QObject::connect(polygon, &GePolygon::polygonChanged, this, [=](const QPolygonF& rect,
						GeObjectId& modifiedObjectId, bool b, GePolygon* originGraphicsItem) {
					emit GraphicsPointerExt::polygonChanged(rect, modifiedObjectId.getObjectId(), this);
				});


				qPolygon.append(QPointF(worldX, worldY));
				polygon->setPolygon(qPolygon);

				polygon->setSelected(true);
        		polygonOn = true;
        	}
        	else {
                auto dx = worldX - m_previousPosition.x();
                auto dy = worldY - m_previousPosition.y();
                GePolygon * polygon = qgraphicsitem_cast<GePolygon *>(currentItem);
                if ( polygonVolatyOn) {
                	qPolygon.replace(qPolygon.length()-1, QPointF(worldX, worldY));
                } else {
                	qPolygon.append(QPointF(worldX, worldY));
                }
                polygonVolatyOn = false;
				polygon->setPolygon(qPolygon);
        	}
        }
        break;
    }
    case PolyLineType: {
        if (m_leftMouseButtonPressed && !(button == Qt::RightButton) &&
        		!(button == Qt::MiddleButton)) {
        	if (!polygonOn) {
        		// First point
        		deselectItems();
        		GeObjectId objectId(1, 1);
        		GePolyLine *polyline = new GePolyLine(/*vaData,*/ globalParameters, objectId, m_currentColor, true);
				currentItem = polyline;
				canvas->addItem(currentItem);
				currentItem->setZValue(m_defaultZ);
				connect(polyline, &GePolyLine::clicked, this, &GraphicsPointerExt::signalSelectItem);
				connect(polyline, &GePolyLine::signalMove, this, &GraphicsPointerExt::slotMove);
				connect(polyline, &GePolyLine::polygonPointAddedSignal, this, &GraphicsPointerExt::polyLinePointAdded);
				connect(polyline, &GePolyLine::polygonPointDeletedSignal, this, &GraphicsPointerExt::polyLinePointDeleted);
				connect(polyline, &GePolyLine::itemSelected, this, &GraphicsPointerExt::objectSelectedFromItem);

				emit signalNewSelectItem(polyline, m_indexColor);
				QObject::connect(polyline, &GePolyLine::polygonChanged, this, [=](const QPolygonF& rect,
						GeObjectId& modifiedObjectId, bool b, GePolyLine* originGraphicsItem) {
					emit GraphicsPointerExt::polyLineChanged(rect, modifiedObjectId.getObjectId(), this);
				});


				qPolygon.append(QPointF(worldX, worldY));
				polyline->setPolygon(qPolygon);

				polyline->setSelected(true);
        		polygonOn = true;
        	}
        	else {
                auto dx = worldX - m_previousPosition.x();
                auto dy = worldY - m_previousPosition.y();
                GePolyLine * polyline = qgraphicsitem_cast<GePolyLine *>(currentItem);
                if ( polygonVolatyOn) {
                	qPolygon.replace(qPolygon.length()-1, QPointF(worldX, worldY));
                } else {
                	qPolygon.append(QPointF(worldX, worldY));
                }
                polygonVolatyOn = false;
				polyline->setPolygon(qPolygon);
        	}
        }
        break;
    }
//    case PenType: {
//        if (m_leftMouseButtonPressed && !(event->button() & Qt::RightButton) && !(event->button() & Qt::MiddleButton)) {
//        }
//        break;
//    }
    case RectangleType: {
        if (m_leftMouseButtonPressed && !(button & Qt::RightButton) && !(button & Qt::MiddleButton)) {
            deselectItems();
            GeObjectId objectId(1, 1);
            GeRectangle *rectangle = new GeRectangle(/*vaData, */globalParameters, ""/*vaData->getObjectName(currentObjectId.getObjectId())*/,
            		objectId, m_currentColor, true, this);
            currentItem = rectangle;
            canvas->addItem(currentItem);
            currentItem->setZValue(m_defaultZ);
            connect(rectangle, &GeRectangle::clicked, this, &GraphicsPointerExt::signalSelectItem);
            connect(rectangle, &GeRectangle::signalMove, this, &GraphicsPointerExt::slotMove);

            connect(rectangle, &GeRectangle::itemSelected, this, &GraphicsPointerExt::objectSelectedFromItem);

            emit signalNewSelectItem(rectangle, m_indexColor);
            QObject::connect(rectangle, &GeRectangle::rectangleChanged, this, [=](const QRectF& rect,
            		int modifiedObjectId, bool b) {
            	emit GraphicsPointerExt::rectangleChanged(rect/*, modifiedObjectId.getObjectId()*/, this);
            });

            rectangle->setSelected(true);
        }
        break;
    }

    case EllipseType: {
    if (m_leftMouseButtonPressed && !(button & Qt::RightButton) && !(button & Qt::MiddleButton)) {
        deselectItems();
        GeObjectId objectId(1, 1);
        GeEllipse *ellipse = new GeEllipse(/*vaData,*/ globalParameters, objectId, m_currentColor, true, this);
        currentItem = ellipse;
        canvas->addItem(currentItem);
        currentItem->setZValue(m_defaultZ);
        connect(ellipse, &GeEllipse::clicked, this, &GraphicsPointerExt::signalSelectItem);
        connect(ellipse, &GeEllipse::signalMove, this, &GraphicsPointerExt::slotMove);

        connect(ellipse, &GeEllipse::itemSelected, this, &GraphicsPointerExt::objectSelectedFromItem);

        emit signalNewSelectItem(ellipse, m_indexColor);
        QObject::connect(ellipse, &GeEllipse::ellipseChanged, this, [=](const QRectF& rect,
        		GeObjectId& modifiedObjectId, bool b) {
        	emit GraphicsPointerExt::ellipseChanged(rect, modifiedObjectId.getObjectId(), this);
        });

        ellipse->setSelected(true);
    }
    break;
    }
    case SelectionType: {
        if (m_leftMouseButtonPressed && !(button & Qt::RightButton) && !(button & Qt::MiddleButton)) {
            deselectItems();
//            VESelectionRect *selection = new VESelectionRect(this);
//            currentItem = selection;
//            canvas->addItem(currentItem);
        }
        break;
    }
    default: {
    	PickingTask::mousePressed(worldX, worldY, button, keys, info);
        break;
    }
    }
}

void GraphicsPointerExt::mouseMoved(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,
		const QVector<PickingInfo> & info) {
    switch (m_currentAction) {
    case PolygonType: {
    	if (polygonOn) {
    		GePolygon * polygon = qgraphicsitem_cast<GePolygon *>(currentItem);
    		if (polygonVolatyOn) {
                qPolygon.replace(qPolygon.length()-1, QPointF(worldX, worldY));
				polygon->setPolygon(qPolygon);
    		}
    		else {
    			qPolygon.append( QPointF(worldX, worldY));
    			polygon->setPolygon(qPolygon);
    		}
    		polygonVolatyOn = true;
    	}
        break;
    }
    case PolyLineType: {
    	if (polygonOn) {
    		GePolyLine * polyline = qgraphicsitem_cast<GePolyLine *>(currentItem);
    		if (polygonVolatyOn) {
                qPolygon.replace(qPolygon.length()-1, QPointF(worldX, worldY));
				polyline->setPolygon(qPolygon);
    		}
    		else {
    			qPolygon.append( QPointF(worldX, worldY));
    			polyline->setPolygon(qPolygon);
    		}
    		polygonVolatyOn = true;
    	}
        break;
    }
//    case PenType: {
//    }
    case RectangleType: {
        if (m_leftMouseButtonPressed) {
            auto dx = worldX - m_previousPosition.x();
            auto dy = worldY - m_previousPosition.y();
            GeRectangle * rectangle = qgraphicsitem_cast<GeRectangle *>(currentItem);
            rectangle->setRect((dx > 0) ? m_previousPosition.x() : worldX,
                               (dy > 0) ? m_previousPosition.y() : worldY,
                               qAbs(dx), qAbs(dy));
        }
        break;
    }

    case EllipseType: {
    if (m_leftMouseButtonPressed) {
        auto dx = worldX - m_previousPosition.x();
        auto dy = worldY - m_previousPosition.y();
        GeEllipse * ellipse = qgraphicsitem_cast<GeEllipse *>(currentItem);
        ellipse->setRect((dx > 0) ? m_previousPosition.x() : worldX,
                           (dy > 0) ? m_previousPosition.y() : worldY,
                           qAbs(dx), qAbs(dy));
    }
    break;
    }
//    case SelectionType: {
//        if (m_leftMouseButtonPressed) {
//        }
//        break;
//    }
    default: {
    	PickingTask::mouseMoved(worldX, worldY, button, keys, info);
        break;
    }
    }
}

void GraphicsPointerExt::mouseRelease(double worldX,double worldY,Qt::MouseButton button,
		Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info)
{
    if (button & Qt::LeftButton) m_leftMouseButtonPressed = false;
    switch (m_currentAction) {
//    case LineType:{
//    }
//    case PenType: {
//        break;
//    }
    case RectangleType: {
        if (!m_leftMouseButtonPressed&&
                !(button & Qt::RightButton) &&
                !(button & Qt::MiddleButton)) {
            GeRectangle * rectangle = qgraphicsitem_cast<GeRectangle *>(currentItem);
            rectangle->setPositionGrabbers();
            emit signalSelectItem(rectangle);
            // Rectangle creation in data
            emit GraphicsPointerExt::rectangleChanged(rectangle->rect(), /*rectangle->getObjectId().getObjectId(),*/ this);

        }
        QColor c(0,0,0,0);
        setCurrentAction2(GraphicsPointerExt::DefaultType, -1, c);
        break;
    }
    case EllipseType: {
		if (!m_leftMouseButtonPressed&&
				!(button & Qt::RightButton) &&
				!(button & Qt::MiddleButton)) {
			GeEllipse * ellipse = qgraphicsitem_cast<GeEllipse *>(currentItem);
			ellipse->setPositionGrabbers();
			emit signalSelectItem(ellipse);
			// Rectangle creation in data
			emit GraphicsPointerExt::ellipseChanged(ellipse->rect(), ellipse->getObjectId().getObjectId(), this);
		}
		break;
    }
//
//    case SelectionType: {
//        break;
//    }
    default: {
    	PickingTask::mouseRelease(worldX, worldY, button, keys, info);
        break;
    }
    }
}

void GraphicsPointerExt::mouseDoubleClick(double worldX,double worldY,Qt::MouseButton button,
		Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info)
{
	switch (m_currentAction) {
	case PolygonType:
	{
		if (polygonOn) {
			// First point
			deselectItems();
			GePolygon * polygon = qgraphicsitem_cast<GePolygon *>(currentItem);
			//polygon->setPositionGrabbers();

			QColor c(0,0,0,0);
			setCurrentAction2(GraphicsPointerExt::DefaultType, -1, c);
			emit signalSelectItem(polygon);
			emit endEditionItem(polygon);
			// Rectangle creation in data
			//       emit GraphicsPointerExt::polygonChanged(polygon->polygon(), polygon->getObjectId().getObjectId());

		} else {
			QColor c(0,0,0,0);
			setCurrentAction2(GraphicsPointerExt::DefaultType, -1, c);
		}
		/////////////
	}
	case PolyLineType:
	{
		if (polygonOn) {
			// First point
			deselectItems();
			GePolyLine * polyline = qgraphicsitem_cast<GePolyLine *>(currentItem);
			//polygon->setPositionGrabbers();

			QColor c(0,0,0,0);
			setCurrentAction2(GraphicsPointerExt::DefaultType, -1, c);
			emit signalSelectItem(polyline);
			emit endEditionItem(polyline);
			// Rectangle creation in data
			//       emit GraphicsPointerExt::polygonChanged(polygon->polygon(), polygon->getObjectId().getObjectId());

		} else {
			QColor c(0,0,0,0);
			setCurrentAction2(GraphicsPointerExt::DefaultType, -1, c);
		}
		/////////////
	}
	break;
	case PenType:
	case RectangleType:
	case EllipseType:
	case SelectionType:
		break;
	default:
		PickingTask::mouseDoubleClick(worldX, worldY, button, keys, info);
		break;
	}
}

//void GraphicsPointerExt::keyPressEvent(QKeyEvent *event)
//{
//	switch (event->key()) {
//    case Qt::Key_Delete: {
//        foreach (QGraphicsItem *item, canvas->selectedItems()) {
//  //          signalRemoveItem(item);
// //           canvas->removeItem(item);
//
//            //TODO question HASSAN    QObject::disconnect(item, &GeRectangle::rectangleChanged, this);
//            //emit itemDeletedSignal(item);
//     //       delete item;
//        }
//        deselectItems();
//        break;
//    }
//    case Qt::Key_A: {
//        if(QApplication::keyboardModifiers() & Qt::ControlModifier){
//            foreach (QGraphicsItem *item, canvas->items()) {
//                item->setSelected(true);
//            }
//            if(canvas->selectedItems().length() == 1) signalSelectItem(canvas->selectedItems().at(0));
//        }
//        break;
//    }
//    default:
//        break;
//    }
//    SceneExtension::keyPressEvent(event);
//}

void GraphicsPointerExt::objectSelectedFromItem(int ojectId) {
	emit GraphicsPointerExt::objectSelectedSignal(ojectId);
}

void GraphicsPointerExt::deselectItems() {
    foreach (QGraphicsItem *item, canvas->selectedItems()) {
        item->setSelected(false);
    }
    canvas->selectedItems().clear();
}

/**
 * Top Down
 */
void GraphicsPointerExt::propagSelectionOnGraphics(int objectId) {

    foreach (QGraphicsItem *item, canvas->items()) {
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

void GraphicsPointerExt::slotMove(QGraphicsItem *signalOwner, qreal dx, qreal dy)
{
//    foreach (QGraphicsItem *item, canvas->selectedItems()) {
//        if(item != signalOwner) item->moveBy(dx,dy);
//    }
}

void GraphicsPointerExt::cleanScene() {
	foreach (QGraphicsItem *item, canvas->items()) {
		GeShape *node = dynamic_cast<GeShape *>(item);
	    if (node)
	    	canvas->removeItem( item );
	}
}

void GraphicsPointerExt::cleanObjectGraphicsOnSlice(int objectId) {
	bool found = false;

	foreach (QGraphicsItem *item, canvas->items()) {
		GeShape* shape = dynamic_cast<GeShape*>(item);
		if (shape) {
			if ( shape->getObjectId().getObjectId() == objectId ) {
				canvas->removeItem( item );
				break;
			}
		 }
	}
}

void GraphicsPointerExt::polygonPointAdded(int objectId, int indexAdded, double factor) {
	//int currentSlice = getViewer()->getSliceToBeAnalyse();
	//vaData->addPointInPolygons(objectId, indexAdded, factor, currentSlice);
	emit polygonPointAddedSignal( objectId, indexAdded, factor, /*currentSlice,*/ this);
}

void GraphicsPointerExt::polygonPointDeleted(int objectId, int index) {
	//vaData->deleteVertexInPolygons(objectId, index);

	emit polygonPointDeletedSignal( objectId );
}

void GraphicsPointerExt::polyLinePointAdded(int objectId, int indexAdded, double factor) {
	//int currentSlice = getViewer()->getSliceToBeAnalyse();
	//vaData->addPointInPolygons(objectId, indexAdded, factor, currentSlice);
	emit polyLinePointAddedSignal( objectId, indexAdded, factor, /*currentSlice,*/ this);
}

void GraphicsPointerExt::polyLinePointDeleted(int objectId, int index) {
	//vaData->deleteVertexInPolygons(objectId, index);

	emit polyLinePointDeletedSignal( objectId );
}

/**
 * After slice change
 */
void GraphicsPointerExt::drawAllGraphicsItems() {
	//SyncViewer2d* view2d = getViewer();
//	QGraphicsView* graphicsView =  view2d->getScene()->views().first();

//	int currentSlice = view2d->getSliceToBeAnalyse();
//	int objectsSize = vaData->getObjectsSize();
	// Boucle sur les objets
//	for (int currentObjectIndex = 0; currentObjectIndex < objectsSize; currentObjectIndex++) {
//		QString currentObjectName = vaData->getObjectName(currentObjectIndex);
//		int sliceMin, sliceMax;
//		vaData->getObjectSliceMinMax(currentObjectIndex, sliceMin, sliceMax);
//		if ( sliceMin > currentSlice || sliceMax < currentSlice) continue;

		switch ( m_drawingType/*vaData->getObjectType(currentObjectIndex)*/ ) {
		case DrawingType::Rectangle :
			{
				rectDisplay(/**currentObjectIndex, currentSlice*/1, 1, false);
			}
			break;
		case DrawingType::Polygon :
			{
				polygonDisplay(/*currentObjectIndex, currentSlice*/1, 1, false);
			}
			break;
		case DrawingType::PolyLine :
			{
				polyLineDisplay(/*currentObjectIndex, currentSlice*/1, 1, false);
			}
			break;
		case DrawingType::Ellipse :
			{
				ellipseDisplay(/*currentObjectIndex, currentSlice*/1, 1, false);
			}
			break;
		}
//	}
}

/**
 * After picking or move
 */
void GraphicsPointerExt::refreshObjectDisplay(int objectId) {
//	SyncViewer2d* view2d = getViewer();
//	QGraphicsView* graphicsView =  view2d->getScene()->views().first();

//	int currentSlice = view2d->getSliceToBeAnalyse();
	// Boucle sur les objets

//	QColor c = vaData->getObjectColor(objectId);
//	QString currentObjectName = vaData->getObjectName(objectId);

//	int sliceMin, sliceMax;
//	vaData->getObjectSliceMinMax(objectId, sliceMin, sliceMax);
//	if ( sliceMin > currentSlice || sliceMax < currentSlice) {
//		cleanObjectGraphicsOnSlice(objectId);
//		return;
//	}

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
	switch ( m_drawingType ) {
	case DrawingType::Rectangle :
		{
			rectDisplay(/**currentObjectIndex, currentSlice*/1, 1, false);
		}
		break;
	case DrawingType::Polygon :
		{
			polygonDisplay(/*currentObjectIndex, currentSlice*/1, 1, false);
		}
		break;
	case DrawingType::PolyLine :
		{
			polyLineDisplay(/*currentObjectIndex, currentSlice*/1, 1, false);
		}
		break;
	case DrawingType::Ellipse :
		{
			ellipseDisplay(/*currentObjectIndex, currentSlice*/1, 1, false);
		}
		break;
	}
}

/**
 * Single object
 */
void GraphicsPointerExt::rectDisplay(int objectId, int currentSlice, bool removeBefore) {
	QRectF currentRect;
//	QColor c = vaData->getObjectColor(objectId);
//	if ( true /*! vaData->findRect(currentSlice, objectId, currentRect)*/) {
		// Object does not exists on this slice
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

/**
 * Single object
 */
void GraphicsPointerExt::polygonDisplay(int objectId, int currentSlice, bool removeBefore) {
//	QPolygonF currentPolygon;
//	QColor c = vaData->getObjectColor(objectId);
//	if ( ! vaData->findPolygon(currentSlice, objectId, currentPolygon)) {
//		// Object does not exists on this slice
//		int beforeIndex, nextIndex;
//		QPolygonF polygonBefore, polygonNext;
//		if ( vaData->findPolygonObjectBeforeAfter(objectId, currentSlice,
//				beforeIndex, polygonBefore, nextIndex, polygonNext)) {
//			if (nextIndex == data::MtVaObject::UNDEF_AFTER_SLICE && beforeIndex < currentSlice) {
//				refreshOrCreateItem(objectId, polygonBefore, c, false, removeBefore );
//			}
//
//			else if (beforeIndex == data::MtVaObject::UNDEF_BEFORE_SLICE  && nextIndex > currentSlice) {
//				refreshOrCreateItem(objectId, polygonNext, c, false, removeBefore );
//
//			} else {
//				QPolygonF newPolygon;
//				for ( int i = 0; i < polygonBefore.size(); i++) {
//					QPointF pul0 = polygonBefore.at(i);
//					QPointF pul1 = polygonNext.at(i);
//					float xuli = ((currentSlice - beforeIndex) * (pul1.x() - pul0.x()) / (nextIndex - beforeIndex)) + pul0.x();
//					float yuli = ((currentSlice - beforeIndex) * (pul1.y() - pul0.y()) / (nextIndex - beforeIndex)) + pul0.y();
//					newPolygon.append(QPointF(xuli, yuli));
//				}
//				refreshOrCreateItem(objectId, newPolygon, c, false, removeBefore );
//			}
//		}
//		else {
//			//	NON TROUVE
//		}
//	}
//	else { // Il y a un rectangle sur cette slice il faut le dessiner
//		refreshOrCreateItem(objectId, currentPolygon, c, true, removeBefore );
//	}
}

/**
 * Single object
 */
void GraphicsPointerExt::polyLineDisplay(int objectId, int currentSlice, bool removeBefore) {
//	QPolygonF currentPolygon;
//	QColor c = vaData->getObjectColor(objectId);
//	if ( ! vaData->findPolygon(currentSlice, objectId, currentPolygon)) {
//		// Object does not exists on this slice
//		int beforeIndex, nextIndex;
//		QPolygonF polygonBefore, polygonNext;
//		if ( vaData->findPolygonObjectBeforeAfter(objectId, currentSlice,
//				beforeIndex, polygonBefore, nextIndex, polygonNext)) {
//			if (nextIndex == data::MtVaObject::UNDEF_AFTER_SLICE && beforeIndex < currentSlice) {
//				refreshOrCreateItem(objectId, polygonBefore, c, false, removeBefore );
//			}
//
//			else if (beforeIndex == data::MtVaObject::UNDEF_BEFORE_SLICE  && nextIndex > currentSlice) {
//				refreshOrCreateItem(objectId, polygonNext, c, false, removeBefore );
//
//			} else {
//				QPolygonF newPolygon;
//				for ( int i = 0; i < polygonBefore.size(); i++) {
//					QPointF pul0 = polygonBefore.at(i);
//					QPointF pul1 = polygonNext.at(i);
//					float xuli = ((currentSlice - beforeIndex) * (pul1.x() - pul0.x()) / (nextIndex - beforeIndex)) + pul0.x();
//					float yuli = ((currentSlice - beforeIndex) * (pul1.y() - pul0.y()) / (nextIndex - beforeIndex)) + pul0.y();
//					newPolygon.append(QPointF(xuli, yuli));
//				}
//				refreshOrCreateItem(objectId, newPolygon, c, false, removeBefore );
//			}
//		}
//		else {
//			//	NON TROUVE
//		}
//	}
//	else { // Il y a un rectangle sur cette slice il faut le dessiner
//		refreshOrCreateItem(objectId, currentPolygon, c, true, removeBefore );
//	}
}

/**
 * Single object
 */
void GraphicsPointerExt::ellipseDisplay(int objectId, int currentSlice, bool removeBefore) {
//	QRectF currentRect;
//	QColor c = vaData->getObjectColor(objectId);
//	if ( ! vaData->findEllipse(currentSlice, objectId, currentRect)) {
//		// Object does not exists on this slice
//		int beforeIndex, nextIndex;
//		QRectF rectBefore(0,0,0,0), rectNext(0,0,0,0) ;
//		if ( vaData->findEllipseObjectBeforeAfter(objectId, currentSlice,
//				beforeIndex, rectBefore, nextIndex, rectNext)) {
//			if (nextIndex == data::MtVaObject::UNDEF_AFTER_SLICE && beforeIndex < currentSlice) {
//				refreshOrCreateEllipseItem(objectId, rectBefore, c, false, removeBefore );
//			}
//
//			else if (beforeIndex == data::MtVaObject::UNDEF_BEFORE_SLICE  && nextIndex > currentSlice) {
//				refreshOrCreateEllipseItem(objectId, rectNext, c, false, removeBefore );
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
//				refreshOrCreateEllipseItem(objectId, interpRect, c, false, removeBefore );
//			}
//		}
//		else {
//			//	NON TROUVE
//		}
//	}
//	else { // Il y a un rectangle sur cette slice il faut le dessiner
//		refreshOrCreateEllipseItem(objectId, currentRect, c, true, removeBefore );
//	}
}

void GraphicsPointerExt::refreshOrCreateItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere, bool removeBefore) {
	bool found = false;

	if ( removeBefore ) {
		foreach (QGraphicsItem *item, canvas->items()) {
			GeShape* shape = dynamic_cast<GeShape*>(item);
			if (shape) {
				if ( shape->getObjectId().getObjectId() == objectId ) {
					canvas->removeItem( item );
					if ( !found ) createItem(objectId, newRect, color, pickedHere);

					found = true;
					break;
				}
			 }
		}
	}
	if (!found) {
		// Create
		createItem( objectId, newRect, color, pickedHere);
	}
}

void GraphicsPointerExt::refreshOrCreateItem(int objectId, QPolygonF& newRect, QColor& color, bool pickedHere, bool removeBefore) {
	bool found = false;

	if ( removeBefore ) {
		foreach (QGraphicsItem *item, canvas->items()) {
			GePolygon* polygon_cast = dynamic_cast<GePolygon*>(item);
			GePolyLine* polyline_cast = dynamic_cast<GePolyLine*>(item);
			if (polygon_cast) {
				if ( polygon_cast->getObjectId().getObjectId() == objectId ) {
					canvas->removeItem( item );
					if ( !found ) createItem(objectId, newRect, color, pickedHere);

					found = true;
					break;
				}
			} else if (polyline_cast) {
				if ( polyline_cast->getObjectId().getObjectId() == objectId ) {
					canvas->removeItem( item );
					if ( !found ) createPolyLineItem(objectId, newRect, color, pickedHere);

					found = true;
					break;
				}
			}
		}
	}
	if (!found) {
		// Create
		createItem( objectId, newRect, color, pickedHere);
	}
}

void GraphicsPointerExt::refreshOrCreateEllipseItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere, bool removeBefore) {
	bool found = false;

	if ( removeBefore ) {
		foreach (QGraphicsItem *item, canvas->items()) {
			GeEllipse* ellipseCast = dynamic_cast<GeEllipse*>(item);
			if (ellipseCast) {
				if ( ellipseCast->getObjectId().getObjectId() == objectId ) {
					canvas->removeItem( item );
					if ( !found ) createEllipseItem(objectId, newRect, color, pickedHere);

					found = true;
					break;
				}
			 }
		}
	}
	if (!found) {
		// Create
		createEllipseItem( objectId, newRect, color, pickedHere);
	}
}

void GraphicsPointerExt::createItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere) {
	GeObjectId oId(objectId, 23);
	GeRectangle *rectangle = new GeRectangle(/*vaData, */globalParameters, /*vaData->getObjectName(objectId)*/"",
			oId, color, pickedHere, newRect, this);
	currentItem = rectangle;
	canvas->addItem(currentItem);
    currentItem->setZValue(m_defaultZ);
	connect(rectangle, &GeRectangle::clicked, this, &GraphicsPointerExt::signalSelectItem);
	connect(rectangle, &GeRectangle::signalMove, this, &GraphicsPointerExt::slotMove);

	connect(rectangle, &GeRectangle::itemSelected, this, &GraphicsPointerExt::objectSelectedFromItem);

	// emit signalNewSelectItem(rectangle, m_indexColor);
	QObject::connect(rectangle, &GeRectangle::rectangleChanged, this, [=](const QRectF& rect,
			int modifiedObjectId, bool b) {
		emit GraphicsPointerExt::rectangleChanged(rect, /*modifiedObjectId.getObjectId(),*/ this);
	});
	QColor c(0,0,0,0);
	setCurrentAction2(GraphicsPointerExt::DefaultType, -1, c);
}

void GraphicsPointerExt::createItem(int objectId, QPolygonF& newPolygon, QColor& color, bool pickedHere) {
	GeObjectId oId(objectId, 23);
	GePolygon *polygon = new GePolygon(/*vaData,*/ globalParameters, oId, color, pickedHere, this);
	polygon->setPolygon(newPolygon);
	currentItem = polygon;
	canvas->addItem(currentItem);
    currentItem->setZValue(m_defaultZ);
	connect(polygon, &GePolygon::clicked, this, &GraphicsPointerExt::signalSelectItem);
	connect(polygon, &GePolygon::signalMove, this, &GraphicsPointerExt::slotMove);
	connect(polygon, &GePolygon::polygonPointAddedSignal, this, &GraphicsPointerExt::polygonPointAdded);
	connect(polygon, &GePolygon::polygonPointDeletedSignal, this, &GraphicsPointerExt::polygonPointDeleted);
	connect(polygon, &GePolygon::itemSelected, this, &GraphicsPointerExt::objectSelectedFromItem);

	// emit signalNewSelectItem(rectangle, m_indexColor);
	QObject::connect(polygon, &GePolygon::polygonChanged, this, [=](const QPolygonF& poly,
			GeObjectId& modifiedObjectId, bool b, GePolygon* originGraphicsItem) {
		emit GraphicsPointerExt::polygonChanged(poly, modifiedObjectId.getObjectId(), this);
	});
	QColor c(0,0,0,0);
	setCurrentAction2(GraphicsPointerExt::DefaultType, -1, c);
}

void GraphicsPointerExt::createPolyLineItem(int objectId, QPolygonF& newPolygon, QColor& color, bool pickedHere) {
	GeObjectId oId(objectId, 23);
	GePolyLine *polyline = new GePolyLine(/*vaData,*/ globalParameters, oId, color, pickedHere, this);
	polyline->setPolygon(newPolygon);
	currentItem = polyline;
	canvas->addItem(currentItem);
    currentItem->setZValue(m_defaultZ);
	connect(polyline, &GePolyLine::clicked, this, &GraphicsPointerExt::signalSelectItem);
	connect(polyline, &GePolyLine::signalMove, this, &GraphicsPointerExt::slotMove);
	connect(polyline, &GePolyLine::polygonPointAddedSignal, this, &GraphicsPointerExt::polyLinePointAdded);
	connect(polyline, &GePolyLine::polygonPointDeletedSignal, this, &GraphicsPointerExt::polyLinePointDeleted);
	connect(polyline, &GePolyLine::itemSelected, this, &GraphicsPointerExt::objectSelectedFromItem);

	// emit signalNewSelectItem(rectangle, m_indexColor);
	QObject::connect(polyline, &GePolyLine::polygonChanged, this, [=](const QPolygonF& poly,
			GeObjectId& modifiedObjectId, bool b, GePolyLine* originGraphicsItem) {
		emit GraphicsPointerExt::polyLineChanged(poly, modifiedObjectId.getObjectId(), this);
	});
	QColor c(0,0,0,0);
	setCurrentAction2(GraphicsPointerExt::DefaultType, -1, c);
}

void GraphicsPointerExt::createEllipseItem(int objectId, QRectF& newRect, QColor& color, bool pickedHere) {
	GeObjectId oId(objectId, 23);
	GeEllipse *rectangle = new GeEllipse(/*vaData,*/ globalParameters, oId, color, pickedHere, newRect, this);
	currentItem = rectangle;
	canvas->addItem(currentItem);
    currentItem->setZValue(m_defaultZ);
	connect(rectangle, &GeEllipse::clicked, this, &GraphicsPointerExt::signalSelectItem);
	connect(rectangle, &GeEllipse::signalMove, this, &GraphicsPointerExt::slotMove);

	connect(rectangle, &GeEllipse::itemSelected, this, &GraphicsPointerExt::objectSelectedFromItem);

	// emit signalNewSelectItem(rectangle, m_indexColor);
	QObject::connect(rectangle, &GeEllipse::ellipseChanged, this, [=](const QRectF& rect,
			GeObjectId& modifiedObjectId, bool b) {
		emit GraphicsPointerExt::ellipseChanged(rect, modifiedObjectId.getObjectId(),this);
	});
	QColor c(0,0,0,0);
	setCurrentAction2(GraphicsPointerExt::DefaultType, -1, c);
}

void GraphicsPointerExt::setSlice(int slice) {
	    qDebug() << "SET SLICE" << slice;
}

void GraphicsPointerExt::setDefaultZ(int newZ) {
	m_defaultZ = newZ;
	if (currentItem!=nullptr) {
		currentItem->setZValue(m_defaultZ);
	}
}
int GraphicsPointerExt::defaultZ() const {
	return m_defaultZ;
}

//void GraphicsPointerExt::initCanvas(QExtendableGraphicsScene* canvas) {
//	canvas->addExtension(this);
//	this->canvas = canvas;
//	//canvas->addItem(&editableRect);
//}
//
//void GraphicsPointerExt::releaseCanvas(QExtendableGraphicsScene* canvas) {
//	canvas->removeExtension(this);
//	//canvas->removeItem(&editableRect);
//}
