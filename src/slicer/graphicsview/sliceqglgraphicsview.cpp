#include "sliceqglgraphicsview.h"
#include "globalconfig.h"
#include "GraphicSceneEditor.h"

#include <QMouseEvent>
#include <cmath>

SliceQGLGraphicsView::SliceQGLGraphicsView(QWidget *parent) :
		BaseQGLGraphicsView(parent) {
	setDragMode(QGraphicsView::NoDrag);
	setAlignment(Qt::AlignLeft | Qt::AlignTop);
}

void SliceQGLGraphicsView::wheelEvent(QWheelEvent *e) {
	if ((!scene()->selectedItems().isEmpty()) &&  (e->modifiers() & Qt::ShiftModifier) )
	{
		QPoint numDegrees = e->angleDelta() / 8;
		if (!numDegrees.isNull())
		{
			if (numDegrees.y() >= 15)
			{
				dynamic_cast<GraphicSceneEditor*>(scene())->rotate(5);
			}
			else if (numDegrees.y() <= -15)
			{
				dynamic_cast<GraphicSceneEditor*>(scene())->rotate(-5);
			}
		}
	}
	else
	{
		if (zoomLocked())
			return;


		int delta = e->angleDelta().y();//manhattanLength();//original e->delta()
		GlobalConfig& config = GlobalConfig::getConfig();
		double factor = std::pow(config.wheelZoomBaseFactor(),config.wheelZoomExponentFactor() * delta);
		setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
		if (Qt::KeyboardModifier::ShiftModifier == e->modifiers())
			applyScale(factor, 1);
		else if (Qt::KeyboardModifier::ControlModifier == e->modifiers())
			applyScale(1, factor);
		else
			applyScale(factor, factor);
	}
}
