/*
 * GraphicSceneEditor.cpp
 *
 *  Created on: Sep 6, 2021
 *      Author: l1046262
 */


#include <QTextCursor>
#include <QGraphicsSceneMouseEvent>
#include <QtGlobal>
#include <QDebug>
#include <QSettings>
#include <QMessageBox>
#include <QFileInfo>
#include <QAction>
#include <boost/filesystem.hpp>

#include "GraphicSceneEditor.h"
#include "GraphEditor_RectShape.h"
#include "GraphEditor_PolygonShape.h"
#include "GraphEditor_CurveShape.h"
#include "GraphEditor_PolyLineShape.h"
#include "GraphEditor_EllipseShape.h"
#include "GraphEditor_RegularBezierPath.h"
#include "GraphEditor_ListBezierPath.h"
#include "GraphEditor_Path.h"
#include "GraphEditor_LineShape.h"
#include "GraphEditor_TextItem.h"
#include "singlesectionview.h"
#include "GraphicTool_GraphicLayer.h"
#include "geotimegraphicsview.h"
#include "workingsetmanager.h"
#include "randomlineview.h"
#include "splittedview.h"
#include "abstractgraphicrep.h"
#include "iCUDAImageClone.h"
#include "rgbqglcudaimageitem.h"
#include "rgblayerrgtrep.h"
#include "basemapsurface.h"
#include "seismic3dabstractdataset.h"

#include "nurbswidget.h"

namespace fs = boost::filesystem;

GraphicSceneEditor::GraphicSceneEditor(Abstract2DInnerView* innerview, QObject *parent)
: m_InnerView(innerview) ,QGraphicsScene(parent)
{
    m_orthoLine = nullptr;
    m_SliceValue = 0;
    createActions();
    createMenus();
    backupUndostack();
}

void GraphicSceneEditor::createMenus()
{

    itemMenu = new QMenu("Item Menu");
    itemMenu->addAction(display);
    itemMenu->addAction(m_Orthogonal);
    itemMenu->addAction(displayInfo);
    itemMenu->addAction(copyAction);
    itemMenu->addAction(cutAction);
    itemMenu->addAction(pasteAction);
    itemMenu->addAction(deleteAction);
    //    itemMenu->addSeparator();
    //    itemMenu->addAction(undoAction);
    //    itemMenu->addAction(redoAction);
    //    itemMenu->addSeparator();
    //    itemMenu->addAction(groupAction);
    //    itemMenu->addAction(ungroupAction);
    itemMenu->addSeparator();
    itemMenu->addAction(path3dAction);

    itemMenu->addAction(nurbs3dAction);

    itemMenu->addAction(cloneAndKeepAction);

    itemMenu->addSeparator();
    itemMenu->addAction(toFrontAction);
    itemMenu->addAction(sendBackAction);
    itemMenu->addSeparator();
    itemMenu->addAction(deselectWellsAction);

}

void GraphicSceneEditor::createActions()
{
    display = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"),
            tr("Display Section"), this);
    connect(display, SIGNAL(triggered()), this, SLOT(CreateRandomView()));

    m_Orthogonal = new QAction(tr("Display Orthogonal Section"), this);
    connect(m_Orthogonal, SIGNAL(triggered()), this, SLOT(createOrgthognalRandomView()));

    displayInfo = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"),
            tr("Display Info"), this);
    connect(displayInfo, SIGNAL(triggered()), this, SLOT(displayItemInfo()));

    toFrontAction = new QAction(QIcon(":/slicer/icons/graphic_tools/bringtofront.png"),
            tr("Bring to &Front"), this);
    connect(toFrontAction, SIGNAL(triggered()), this, SLOT(bringToFront()));

    sendBackAction = new QAction(QIcon(":/slicer/icons/graphic_tools/sendtoback.png"), tr("Send to &Back"), this);
    connect(sendBackAction, SIGNAL(triggered()), this, SLOT(sendToBack()));

    deleteAction = new QAction(QIcon(":/slicer/icons/graphic_tools/delete.png"), tr("&Delete"), this);
    connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteItem()));

    copyAction = new QAction(QIcon(":/slicer/icons/graphic_tools/copy.png"), tr("C&opy"), this);
    connect(copyAction, SIGNAL(triggered()), this, SLOT(copyItem()));

    pasteAction = new QAction(QIcon(":/slicer/icons/graphic_tools/paste.png"), tr("P&aste"), this);
    connect(pasteAction, SIGNAL(triggered()), this, SLOT(pasteItem()));

    cutAction = new QAction(QIcon(":/slicer/icons/graphic_tools/cut.png"), tr("C&ut"), this);
    connect(cutAction, SIGNAL(triggered()), this, SLOT(cutItem()));

    deselectWellsAction = new QAction(QIcon(":/slicer/icons/graphic_tools/cut_.png"), tr("Deselect Wells"), this);
    connect(deselectWellsAction, SIGNAL(triggered()), this, SLOT(deselectWells()));


	nurbs3dAction = new QAction(QIcon(":/slicer/icons/graphic_tools/bspline.png"), tr("&Nurbs3d"), this);
	connect(nurbs3dAction, SIGNAL(triggered()), this, SLOT(nurbs3d()));

	//createNurbsAction = new QAction(QIcon(":/slicer/icons/graphic_tools/bspline.png"), tr("&Create Nurbs"), this);
	//connect(createNurbsAction, SIGNAL(triggered()), this, SLOT(createNurbs()));

    path3dAction = new QAction(QIcon(":/slicer/icons/graphic_tools/bezier.png"), tr("&Path3d"), this);
    connect(path3dAction, SIGNAL(triggered()), this, SLOT(path3d()));


    cloneAndKeepAction = new QAction(QIcon(":/slicer/icons/graphic_tools/_.png"), tr("&Clone and Keep"), this);
    connect(cloneAndKeepAction, SIGNAL(triggered()), this, SLOT(cloneAndKeep()));

    //    undoAction = new QAction(QIcon(":images/undo.png"), tr("U&ndo"), this);
    //    connect(undoAction, SIGNAL(triggered()), this, SLOT(undo()));
    //
    //    redoAction = new QAction(QIcon(":images/redo.png"), tr("R&edo"), this);
    //    connect(redoAction, SIGNAL(triggered()), this, SLOT(redo()));
    //
    //    groupAction = new QAction(QIcon(":images/group.png"), tr("G&roup"), this);
    //    connect(groupAction, SIGNAL(triggered()), this, SLOT(groupItems()));
    //
    //    ungroupAction = new QAction(QIcon(":images/ungroup.png"), tr("U&ngroup"), this);
    //    connect(ungroupAction, SIGNAL(triggered()), this, SLOT(ungroupItems()));

}

void GraphicSceneEditor::mousePressEvent(QGraphicsSceneMouseEvent *mouseEvent)
{

    st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();

    if (st_GraphicSettings.enabled)
    {
        GraphicToolsWidget::setActiveInnerView(m_InnerView->title(), this);


        QPen selectedPen = st_GraphicSettings.pen;
        QBrush selectedBrush = st_GraphicSettings.brush;
        if (st_GraphicSettings.action == eGraphicAction_Draw)
        {


            innerView()->view()->setDragMode(QGraphicsView::NoDrag);

            static int ZValue = 4000;
            eShape shape = st_GraphicSettings.shape;
            int smoothValue = st_GraphicSettings.smooth;
            // No draw in progress
            if (m_EndPicking==1){
                foreach(QGraphicsItem* p, selectedItems()) p->setSelected(false);
                PointsVec.clear();
                PointsVec.push_back(mouseEvent->scenePos());
                m_item = createGraphicsItem(shape, selectedPen, selectedBrush, smoothValue, true);
                m_item->setZValue(ZValue);
                m_item->setSelected(true);
                m_item->setFlag(QGraphicsItem::ItemSendsGeometryChanges,true);
                m_item->setFlag(QGraphicsItem::ItemIsMovable, true );
                ZValue++;
                addItem(m_item);
                m_EndPicking =0;
            }
            // Draw in progress
            else if (!m_EndPicking)
            {
                if ( (shape == eShape_Line) && (PointsVec.size() == 2) )
                {
                    m_EndPicking = 1;
                }
                else if ((shape == eShape_Triangle) && (PointsVec.size() == 3) )
                {
                    m_EndPicking = 1;
                }
                else if ((shape == eShape_ListBezierPath) )
				{
					if (dynamic_cast<GraphEditor_ListBezierPath*>(m_item)->checkClosedPath())
					{

						dynamic_cast<GraphEditor_ListBezierPath*>(m_item)->setDrawFinished(true);
						emit sendDirectriceOk();
						m_EndPicking = 1;
					}
				}
                else if ((shape == eShape_Polyline) || ISCUREVESHAPE(shape))
                {
                    if (dynamic_cast<GraphEditor_PolyLineShape*>(m_item)->checkClosedPath())
                    {
                        dynamic_cast<GraphEditor_PolyLineShape*>(m_item)->setDrawFinished(true);
                        m_EndPicking = 1;
                    }
                }

                if (m_EndPicking != 1)
                {
                    PointsVec.push_back(mouseEvent->scenePos());
                    if (shape == eShape_Polyline)
                    {
                        dynamic_cast<GraphEditor_PolyLineShape*>(m_item)->setPolygon(PointsVec);
                    }
                    else if (shape == eShape_ListBezierPath)
					{

						dynamic_cast<GraphEditor_ListBezierPath*>(m_item)->setPolygon(PointsVec);
					}
                    else if (ISCUREVESHAPE(shape))
                    {
                        dynamic_cast<GraphEditor_CurveShape*>(m_item)->setPolygon(PointsVec);
                    }
                    else if (shape == eShape_Polygon)
                    {
                        dynamic_cast<GraphEditor_PolygonShape*>(m_item)->setPolygon(PointsVec);
                    }
                }
                else if (m_EndPicking == 1)
                {
                    saveItem();
                }
            }
        }
        else if (st_GraphicSettings.action == eGraphicAction_Text)
        {
            GraphEditor_TextItem *textItem = new GraphEditor_TextItem();
            textItem->setPos(mouseEvent->scenePos());
            textItem->setDefaultTextColor(st_GraphicSettings.textColor);
            textItem->setFont(st_GraphicSettings.font);
            textItem->setZValue(6000);
            addItem(textItem);
            textItem->setSelected(true);
            textItem->setFocus();
            m_GraphicItems[m_SliceValue].append(textItem);
            backupUndostack();
        }
        else if (st_GraphicSettings.action == eGraphicAction_Fill){
            m_EndPicking=1;
            foreach(QGraphicsItem* p, selectedItems()) {
                //if (! dynamic_cast<GraphEditor_PolyLineShape*>(p) )
                {
                    if (dynamic_cast<GraphEditor_LineShape *>(p)){
                        dynamic_cast<GraphEditor_LineShape*>(p)->setPen(st_GraphicSettings.pen);
                    }else if (dynamic_cast<QAbstractGraphicsShapeItem*>(p)){
                        dynamic_cast<QAbstractGraphicsShapeItem*>(p)->setPen(st_GraphicSettings.pen);
                        dynamic_cast<QAbstractGraphicsShapeItem*>(p)->setBrush(st_GraphicSettings.brush);
                    }
                }
                //else
                //dynamic_cast<GraphEditor_PolyLineShape*>(p)->setPen(st_GraphicSettings.pen);
            }
        }

        if (st_GraphicSettings.action != eGraphicAction_Draw){
            GraphicToolsWidget::setDefaultAction();
        }

        if (m_EndPicking){
            QGraphicsScene::mousePressEvent(mouseEvent);
        }
        else
        	mouseEvent->accept();
    }else{
        QGraphicsScene::mousePressEvent(mouseEvent);
    }
}

void GraphicSceneEditor::mouseMoveEvent(QGraphicsSceneMouseEvent *mouseEvent)
{

    st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();

    if (st_GraphicSettings.enabled){
        eShape shape = st_GraphicSettings.shape;
        if (st_GraphicSettings.action == eGraphicAction_Draw){
            if (!m_EndPicking){
                if ( (PointsVec.size()>1) && ((shape != eShape_FreeDraw) &&  (shape != eShape_RegularBezierPath))){
                    PointsVec.erase(PointsVec.end()-1);
                }
                PointsVec.push_back(mouseEvent->scenePos());

                if (ISRECTSHAPE(shape)){
                    dynamic_cast<GraphEditor_RectShape*>(m_item)->setRect(CreateRect(PointsVec[0],PointsVec[1]));
                }
                else if (shape == eShape_Line){
                    dynamic_cast<GraphEditor_LineShape*>(m_item)->setLine(QLineF(PointsVec[0],PointsVec[1]));
                }
                else if (shape == eShape_Ellipse){
                    dynamic_cast<GraphEditor_EllipseShape*>(m_item)->setRect(CreateRect(PointsVec[0],PointsVec[1]));
                }
                else if (shape == eShape_Circle){
                    QRectF r = CreateRect(PointsVec[0],PointsVec[1]);
                    int medium = (r.width() + r.height())/2;
                    r = QRectF(r.topLeft().x(),r.topLeft().y(), medium, medium);
                    dynamic_cast<GraphEditor_EllipseShape*>(m_item)->setRect(r);
                }
                else if (shape == eShape_Polyline){
                    dynamic_cast<GraphEditor_PolyLineShape*>(m_item)->setPolygon(PointsVec);
                }
                else if (shape == eShape_ListBezierPath){
                	GraphEditor_ListBezierPath* path = dynamic_cast<GraphEditor_ListBezierPath*>(m_item);
					dynamic_cast<GraphEditor_ListBezierPath*>(m_item)->setPolygon(PointsVec);
				}
                else if (ISCUREVESHAPE(shape)){

                    dynamic_cast<GraphEditor_CurveShape*>(m_item)->setPolygon(PointsVec);
                }
                else if (shape == eShape_RegularBezierPath){
                    dynamic_cast<GraphEditor_Path*>(m_item)->setPolygon(PointsVec);
                }

                else if (shape == eShape_FreeDraw){
                    dynamic_cast<GraphEditor_Path*>(m_item)->setPolygon(PointsVec);
                }else {
                    dynamic_cast<GraphEditor_PolygonShape*>(m_item)->setPolygon(PointsVec);
                }
            }
        }
        if (m_EndPicking)
        {

            QGraphicsScene::mouseMoveEvent(mouseEvent);
        }
    }
    else
    {
        QGraphicsScene::mouseMoveEvent(mouseEvent);
    }
}

void GraphicSceneEditor::mouseReleaseEvent(QGraphicsSceneMouseEvent *mouseEvent)
{

    st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();
    if (st_GraphicSettings.enabled)
    {
        if (st_GraphicSettings.action == eGraphicAction_Draw)
        {

            eShape shape = st_GraphicSettings.shape;

            if (shape == eShape_Rect || shape == eShape_RoundRect
                    || shape == eShape_Circle || shape == eShape_Ellipse
                    || shape == eShape_FreeDraw || shape == eShape_RegularBezierPath  )
            {

            	if (shape == eShape_Circle || shape == eShape_Ellipse)
				{
            		GraphEditor_EllipseShape * circle = dynamic_cast<GraphEditor_EllipseShape *> (m_item);

            		connect(this,SIGNAL(updateWidthRandom(int,int)),circle,SLOT(polygonResize(int,int)));
				}
                if (shape == eShape_RegularBezierPath)
                {

                	GraphEditor_RegularBezierPath * bezier = dynamic_cast<GraphEditor_RegularBezierPath *> (m_item);
                	bezier->setDrawFinished(true);
                //	bezier->setInitialWidth(m_randomWidth);
                  //  connect(this,SIGNAL(updateWidthRandom(int,int)),bezier,SLOT(polygonResize(int,int)));

                }
               /* if (shape == eShape_ListBezierPath)
			   {

                	//GraphEditor_ListBezierPath * bezier = dynamic_cast<GraphEditor_ListBezierPath *> (m_item);
                	//->setDrawFinished(true);

                	qDebug()<<" je suis ici";

				 //  connect(this,SIGNAL(updateWidthRandom(int,int)),bezier,SLOT(polygonResize(int,int)));

			   }*/
                /*else if (shape == eShape_CubicBSpline)
				{
                qDebug()<<"je suis ici........";
					GraphEditor_PolyLineShape * bezier = dynamic_cast<GraphEditor_PolyLineShape *> (m_item);

					//bezier->setInitialWidth(m_randomWidth);
					connect(this,SIGNAL(updateWidthRandom(int,int)),bezier,SLOT(polygonResize(int,int)));

				}*/
                else if (shape == eShape_FreeDraw)
                {
                    dynamic_cast<GraphEditor_Path *> (m_item)->setDrawFinished(true);

                   // qDebug() << " position 0 :"<< dynamic_cast<GraphEditor_Path *> (m_item)->
                }
                if (((shape == eShape_RegularBezierPath)||(shape == eShape_FreeDraw))
                        && (PointsVec.size()<3))
                {
                    m_EndPicking =1;

                }
                else
                {

                    saveItem();
                }
            }
        }
        // Detect items moved or resized
        else if (st_GraphicSettings.action == eGraphicAction_Default)
        {
            foreach(QGraphicsItem* p, selectedItems()) {
                if (dynamic_cast<GraphEditor_Item *> (p))
                {
                    if (dynamic_cast<GraphEditor_Item *> (p)->isResized())
                    {
                        dynamic_cast<GraphEditor_Item *> (p)->setResized(false);
                        backupUndostack();
                    }
                    else if (dynamic_cast<GraphEditor_Item *> (p)->isMoved())
                    {
                        dynamic_cast<GraphEditor_Item *> (p)->setMoved(false);
                        GraphEditor_Path * bezier = dynamic_cast<GraphEditor_Path *> (m_item);
                        if(bezier != nullptr)
                        {

                        	moveFinish(bezier);
                        }
                        backupUndostack();
                    }
                }
            }
            foreach (QGraphicsItem* p, items() ) {
                if (dynamic_cast<GraphEditor_TextItem *> (p))
                {
                    if (dynamic_cast<GraphEditor_TextItem *> (p)->contentIsUpdated())
                    {
                        dynamic_cast<GraphEditor_TextItem *> (p)->setUpdated();
                        backupUndostack();
                    }

                    if (dynamic_cast<GraphEditor_TextItem *> (p)->positionIsUpdated())
                    {
                        dynamic_cast<GraphEditor_TextItem *> (p)->setUpdated();
                        backupUndostack();
                    }
                }
            }
        }
    }
    QGraphicsScene::mouseReleaseEvent(mouseEvent);
}

void GraphicSceneEditor::moveFinish(GraphEditor_Path* bezier)
{

	GraphEditor_RegularBezierPath * bezierpath = dynamic_cast<GraphEditor_RegularBezierPath *> (bezier);
	if(bezierpath!= nullptr )
	{
		bezierpath->polygonChanged1();
	}
	else
	{
		GraphEditor_PolyLineShape * polypath = dynamic_cast<GraphEditor_PolyLineShape *> (bezier);
		if(polypath!= nullptr )
		{
			polypath->polygonChanged1();
		}
		else
		{
			GraphEditor_ListBezierPath * listpath = dynamic_cast<GraphEditor_ListBezierPath *> (bezier);
			if(listpath!= nullptr )
			{
				listpath->polygonChanged1();
			}
		}
	}

}

void GraphicSceneEditor::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *mouseEvent) {
    st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();
    if (st_GraphicSettings.enabled)
    {
        GraphicToolsWidget::setActiveInnerView(m_InnerView->title(), this);
        eShape shape = st_GraphicSettings.shape;
        if ((st_GraphicSettings.action == eGraphicAction_Draw) && (PointsVec.size()>2) &&
                ((shape == eShape_Polyline) || (ISCUREVESHAPE(shape))
                        || (shape == eShape_Polygon)) )
        {


        	if ((shape == eShape_ListBezierPath)  )
			{

        		GraphEditor_ListBezierPath* bezierr = dynamic_cast<GraphEditor_ListBezierPath *> (m_item);
				bezierr->setDrawFinished(true);

				bezierr->setSelected(true);

				m_bezierSelected= bezierr;
        		connect(this,SIGNAL(updateWidthRandom(int,int)),bezierr,SLOT(polygonResize(int,int)));
        		emit sendDirectriceOk();


				if ( (PointsVec.last() == PointsVec[PointsVec.size() -2]) )
				{
					PointsVec.removeLast();
					dynamic_cast<GraphEditor_ListBezierPath *> (m_item)->setPolygon(PointsVec);
					dynamic_cast<GraphEditor_ListBezierPath *> (m_item)->setDrawFinished(true);





				}
				else
				{

					//GraphEditor_PolyLineShape * bezier = dynamic_cast<GraphEditor_PolyLineShape *> (m_item);
					//if(bezier != nullptr)
					//	connect(this,SIGNAL(updateWidthRandom(int,int)),bezier,SLOT(polygonResize(int,int)));
				}
			}
        	else if ((shape == eShape_Polyline) || (ISCUREVESHAPE(shape)) )
            {
                if ( (PointsVec.last() == PointsVec[PointsVec.size() -2]) )
                {
                    PointsVec.removeLast();
                    dynamic_cast<GraphEditor_PolyLineShape *> (m_item)->setPolygon(PointsVec);
                    dynamic_cast<GraphEditor_PolyLineShape *> (m_item)->setDrawFinished(true);
                }
                else
                {

                	GraphEditor_PolyLineShape * bezier = dynamic_cast<GraphEditor_PolyLineShape *> (m_item);
                	if(bezier != nullptr)
                		connect(this,SIGNAL(updateWidthRandom(int,int)),bezier,SLOT(polygonResize(int,int)));
                }
            }
            else if ((shape == eShape_ListBezierPath)  )
			{
				if ( (PointsVec.last() == PointsVec[PointsVec.size() -2]) )
				{
					PointsVec.removeLast();
					dynamic_cast<GraphEditor_ListBezierPath *> (m_item)->setPolygon(PointsVec);
					dynamic_cast<GraphEditor_ListBezierPath *> (m_item)->setDrawFinished(true);
					//dynamic_cast<GraphEditor_ListBezierPath *> (m_item)->reinitSelect(true);
				}
				else
				{

					//GraphEditor_PolyLineShape * bezier = dynamic_cast<GraphEditor_PolyLineShape *> (m_item);
					//if(bezier != nullptr)
					//	connect(this,SIGNAL(updateWidthRandom(int,int)),bezier,SLOT(polygonResize(int,int)));
				}
			}
            else if (shape == eShape_Polygon)
            {
                if ( (PointsVec.last() == PointsVec[PointsVec.size() -2]) )
                {
                    PointsVec.removeLast();
                    dynamic_cast<GraphEditor_PolygonShape *> (m_item)->setPolygon(PointsVec);
                }
                if (PointsVec.size()<3)
                {
                    return;
                }
            }
            saveItem();
        }
        else
            QGraphicsScene::mouseDoubleClickEvent(mouseEvent);
    }
    else
        QGraphicsScene::mouseDoubleClickEvent(mouseEvent);
}

void GraphicSceneEditor::keyPressEvent(QKeyEvent *event)
{
    if(event->type() == QKeyEvent::KeyPress)
    {
        if(event->matches(QKeySequence::Copy))
        {
            copyItem();
        }
        else if(event->matches(QKeySequence::Paste))
        {
            pasteItem();
        }
        else if(event->matches(QKeySequence::Cut))
        {
            cutItem();
        }
        else if(event->matches(QKeySequence::Delete))
        {
            deleteItem();
        }
        else if(event->matches(QKeySequence::Find))
        {
            bringToFront();
        }
        else if(event->matches(QKeySequence::AddTab))
        {
            sendToBack();
        }
        //                else if ( (event->key() == Qt::Key_R)  /*&& (event->modifiers() & Qt::ControlModifier)*/){
        //                    redo();
        //                }
        //                else if ((event->key() == Qt::Key_O) /*&& (event->modifiers() & Qt::ControlModifier)*/){
        //                    undo();
        //                }
    }
    QGraphicsScene::keyPressEvent(event);
}

QGraphicsItem* GraphicSceneEditor::createGraphicsItem(eShape shape, QPen selectedPen, QBrush selectedBrush, int smoothValue, bool antialiasingEnabled)
{
    QGraphicsItem *pItem;


    if (shape == eShape_Rect)
    {
        pItem = new GraphEditor_RectShape(CreateRect(PointsVec[0],PointsVec[0]),selectedPen,selectedBrush,itemMenu);
    }
    else if (shape == eShape_RoundRect)
    {
        pItem = new GraphEditor_RectShape(CreateRect(PointsVec[0],PointsVec[0]),selectedPen,selectedBrush,itemMenu, true);
    }
    else if (shape == eShape_Circle || shape == eShape_Ellipse)
    {
        pItem = new GraphEditor_EllipseShape(CreateRect(PointsVec[0],PointsVec[0]),selectedPen,selectedBrush, itemMenu);
    }
    else if (shape == eShape_Polyline)
    {
        pItem = new GraphEditor_PolyLineShape(PointsVec,selectedPen, selectedBrush, itemMenu, this);
    }
    else if (shape == eShape_ListBezierPath)
       {
    	m_colorCustom = NurbsWidget::getCurrentColor();
    	   pItem = new GraphEditor_ListBezierPath(PointsVec,selectedPen, selectedBrush, itemMenu, this,false,m_colorCustom);

       }
    else if (ISCUREVESHAPE(shape))
    {
        pItem = new GraphEditor_CurveShape(PointsVec, shape, selectedPen, selectedBrush, itemMenu, this);
    }
    else if (shape == eShape_RegularBezierPath)
    {
        pItem = new GraphEditor_RegularBezierPath(PointsVec,selectedPen, selectedBrush, smoothValue, itemMenu);
    }

    else if (shape == eShape_FreeDraw)
    {
        pItem = new GraphEditor_Path(PointsVec,selectedPen, selectedBrush, itemMenu);
    }
    else if (shape == eShape_Line)
    {
        QLineF line = QLineF(PointsVec[0],PointsVec[0]);
        pItem = new GraphEditor_LineShape(line ,selectedPen, itemMenu,this);
    }
    else if  (shape == eShape_Triangle)
    {
        pItem = new GraphEditor_PolygonShape(PointsVec,selectedPen,selectedBrush, itemMenu, false);
    }
    else
    {
        pItem = new GraphEditor_PolygonShape(PointsVec,selectedPen,selectedBrush, itemMenu);
    }
    return pItem;
}


void GraphicSceneEditor::setColorCustom(QColor c)
{
	m_colorCustom = c;
}

void GraphicSceneEditor::setGeneratriceColor(QColor color ,QString name)
{

	GraphEditor_ListBezierPath* path = dynamic_cast<GraphEditor_ListBezierPath*>(getCurrentBezier(name));
	if(path!= nullptr)
	{
		path->setColor(color);
	}

}

QColor GraphicSceneEditor::getColorCustom()
{
	return m_colorCustom;
}

void GraphicSceneEditor::createListBezierPath(QString name,QVector<PointCtrl> listepts,QColor col,bool isopen)
{


	 QPen selectedPen(col, 3, Qt::DashLine);
	 selectedPen.setCosmetic(true);
	 QBrush selectedBrush (col);
	 GraphEditor_ListBezierPath *item = new GraphEditor_ListBezierPath(listepts,selectedPen, selectedBrush, itemMenu,this,!isopen,col);
	 item->setZValue(5000);
	// GraphEditor_PolyLineShape* polyline = dynamic_cast<GraphEditor_PolyLineShape*>(item);
	// connect(item,SIGNAL(polygonChanged(QVector<QPointF>,bool)),this,SLOT(onPolygonChanged(QVector<QPointF>,bool)));
//	connect(item,SIGNAL(polygonChanged(QVector<PointCtrl>, QVector<QPointF>,bool)),this, SLOT(onPolygonChangedTangent(QVector<PointCtrl>,QVector<QPointF>,bool)));
	connect(item,SIGNAL(polygonChanged2(GraphEditor_ListBezierPath*)),this, SLOT(onPolygonChangedTangent(GraphEditor_ListBezierPath*)));

	connect(this,SIGNAL(updateWidthRandom(int,int)),item,SLOT(polygonResize(int,int)));

	 addItem(item);
	 m_GraphicItems[m_SliceValue].append(item);



	 item->setSelected(true);


	 createOrgthognalRandomView(name);
}

void GraphicSceneEditor::clearDirectrice(QString name)
{


	if (((m_InnerView->viewType() == BasemapView ) || (m_InnerView->viewType() == StackBasemapView ))&& (m_InnerView->geotimeView() != nullptr)){

		        GeotimeGraphicsView* pGView = dynamic_cast<GeotimeGraphicsView*>(m_InnerView->geotimeView());
		        if(pGView != nullptr){
		            foreach (AbstractInnerView *p , pGView->getInnerViews()){
		                if (p == m_InnerView)
		                {
		                    continue;
		                }
		                if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
		                {
		                    QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
		                    if (scene)
		                    {
		                    	GraphicSceneEditor* sceneCurr = dynamic_cast<GraphicSceneEditor *>(scene);
		                    	GraphEditor_ListBezierPath* bezier = dynamic_cast<GraphEditor_ListBezierPath*>(sceneCurr->getCurrentBezier(name));
		                    	if(bezier!= nullptr)
		                    	{

		                    		sceneCurr->deleteMyItem(m_SliceValue,bezier);
		                    	}


		                    	//dynamic_cast<GraphicSceneEditor *>(scene)->addItem(qgraphicsitem_cast<GraphEditor_ListBezierPath*>(thepath)->clone());

		                    }
		                }
		            }
		        }else {
		            SplittedView* pGView = dynamic_cast<SplittedView*>(m_InnerView->geotimeView());
		            if(pGView != nullptr){
		                foreach (AbstractInnerView *p , pGView->getInnerViews()){
		                    if (p == m_InnerView)
		                    {
		                        continue;
		                    }
		                    if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
		                    {
		                        QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
		                        if (scene)
		                        {
		                        	GraphicSceneEditor* sceneCurr = dynamic_cast<GraphicSceneEditor *>(scene);
									GraphEditor_ListBezierPath* bezier = dynamic_cast<GraphEditor_ListBezierPath*>(sceneCurr->getCurrentBezier(name));
									if(bezier!= nullptr)
									{

										sceneCurr->deleteMyItem(m_SliceValue,bezier);
									}
		                        	//GraphEditor_ListBezierPath* cloned = dynamic_cast<GraphEditor_ListBezierPath*>(thepath)->clone();
									//connect(cloned,SIGNAL(polygonChanged2(GraphEditor_ListBezierPath*)),this, SLOT(onPolygonChangedTangent(GraphEditor_ListBezierPath*)));

									//dynamic_cast<GraphicSceneEditor *>(scene)->addItem(qgraphicsitem_cast<GraphEditor_ListBezierPath*>(cloned));
									//dynamic_cast<GraphicSceneEditor *>(scene)->addItem(qgraphicsitem_cast<GraphEditor_ListBezierPath*>(thepath)->clone());
		                        	//dynamic_cast<GraphicSceneEditor *>(scene)->cloneItem(thepath);
		                           // dynamic_cast<GraphicSceneEditor *>(scene)->saveItemFromOtherScene(m_GraphicItems[m_SliceValue],undoStack);
		                        }
		                    }
		                }
		            }
		        }
		    }

	GraphEditor_ListBezierPath* bezier1 = dynamic_cast<GraphEditor_ListBezierPath*>(getCurrentBezier(name));
		if(bezier1!= nullptr)
		{

			deleteMyItem(m_SliceValue, bezier1);
		}
}


void GraphicSceneEditor::applyColor(QString nameNurbs,QColor col)
{


	if (((m_InnerView->viewType() == BasemapView ) || (m_InnerView->viewType() == StackBasemapView ))&& (m_InnerView->geotimeView() != nullptr)){

	        GeotimeGraphicsView* pGView = dynamic_cast<GeotimeGraphicsView*>(m_InnerView->geotimeView());
	        if(pGView != nullptr){
	            foreach (AbstractInnerView *p , pGView->getInnerViews()){
	                if (p == m_InnerView)
	                {
	                    continue;
	                }
	                if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
	                {
	                    QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
	                    if (scene)
	                    {

	                    	GraphEditor_ListBezierPath* path = dynamic_cast<GraphEditor_ListBezierPath*>( dynamic_cast<GraphicSceneEditor *>(scene)->getCurrentBezier(nameNurbs));
	                    	if(path!= nullptr)
	                    	{

	                    		path->setColor(col);
	                    	}
	                       // dynamic_cast<GraphicSceneEditor *>(scene)->saveItemFromOtherScene(m_GraphicItems[m_SliceValue],undoStack);
	                    }
	                }
	            }
	        }else {
	            SplittedView* pGView = dynamic_cast<SplittedView*>(m_InnerView->geotimeView());
	            if(pGView != nullptr){
	                foreach (AbstractInnerView *p , pGView->getInnerViews()){
	                    if (p == m_InnerView)
	                    {
	                        continue;
	                    }
	                    if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
	                    {
	                        QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
	                        if (scene)
	                        {
	                          //  dynamic_cast<GraphicSceneEditor *>(scene)->saveItemFromOtherScene(m_GraphicItems[m_SliceValue],undoStack);
	                        }
	                    }
	                }
	            }
	        }
	    }
}

void GraphicSceneEditor::cloneDirectrice(GraphEditor_Path * thepath)
{

	/* if (((m_InnerView->viewType() == BasemapView ) || (m_InnerView->viewType() == StackBasemapView ))&& (m_InnerView->geotimeView() != nullptr)){

	        GeotimeGraphicsView* pGView = dynamic_cast<GeotimeGraphicsView*>(m_InnerView->geotimeView());
	        if(pGView != nullptr){
	            foreach (AbstractInnerView *p , pGView->getInnerViews()){
	                if (p == m_InnerView)
	                {
	                    continue;
	                }
	                if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
	                {
	                    QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
	                    if (scene)
	                    {


	                    	GraphicSceneEditor* sceneCurr = dynamic_cast<GraphicSceneEditor *>(scene);
	                    	GraphEditor_ListBezierPath* cloned = dynamic_cast<GraphEditor_ListBezierPath*>(thepath)->clone();
	                    	qDebug()<<p->defaultTitle()<<" addddd item......... "<<cloned->getNameNurbs();
	                    	connect(cloned,SIGNAL(polygonChanged2(GraphEditor_ListBezierPath*)),sceneCurr, SLOT(onPolygonChangedTangent(GraphEditor_ListBezierPath*)));
	                    	connect(cloned,SIGNAL(BezierDeleted(QString)),sceneCurr, SLOT(onDeleted(QString)));
	                    	sceneCurr->addItem(qgraphicsitem_cast<GraphEditor_ListBezierPath*>(cloned));
	                    	sceneCurr->m_GraphicItems[m_SliceValue].append(cloned);


	                    	//dynamic_cast<GraphicSceneEditor *>(scene)->addItem(qgraphicsitem_cast<GraphEditor_ListBezierPath*>(thepath)->clone());

	                    }
	                }
	            }
	        }else {
	            SplittedView* pGView = dynamic_cast<SplittedView*>(m_InnerView->geotimeView());
	            if(pGView != nullptr){
	                foreach (AbstractInnerView *p , pGView->getInnerViews()){
	                    if (p == m_InnerView)
	                    {
	                        continue;
	                    }
	                    if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
	                    {
	                        QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
	                        if (scene)
	                        {
	                        	GraphicSceneEditor* sceneCurr = dynamic_cast<GraphicSceneEditor *>(scene);
								GraphEditor_ListBezierPath* cloned = dynamic_cast<GraphEditor_ListBezierPath*>(thepath)->clone();
								connect(cloned,SIGNAL(polygonChanged2(GraphEditor_ListBezierPath*)),sceneCurr, SLOT(onPolygonChangedTangent(GraphEditor_ListBezierPath*)));
								connect(cloned,SIGNAL(BezierDeleted(QString)),sceneCurr, SLOT(onDeleted(QString)));
								dynamic_cast<GraphicSceneEditor *>(scene)->addItem(qgraphicsitem_cast<GraphEditor_ListBezierPath*>(cloned));
								sceneCurr->m_GraphicItems[m_SliceValue].append(cloned);
								//dynamic_cast<GraphicSceneEditor *>(scene)->addItem(qgraphicsitem_cast<GraphEditor_ListBezierPath*>(thepath)->clone());
	                        	//dynamic_cast<GraphicSceneEditor *>(scene)->cloneItem(thepath);
	                           // dynamic_cast<GraphicSceneEditor *>(scene)->saveItemFromOtherScene(m_GraphicItems[m_SliceValue],undoStack);
	                        }
	                    }
	                }
	            }
	        }
	    }*/
	//  st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();
// st_GraphicSettings.action == eGraphicAction_Draw;
	// m_EndPicking =0;
/*	GraphEditor_ListBezierPath* path = dynamic_cast<GraphEditor_ListBezierPath*>(thepath);

	if(path == nullptr )return;
	//qDebug()<<"GraphicSceneEditor  create directrice :"<<name;
	 QPen selectedPen(path->getColor(), 3, Qt::DashLine);
	 selectedPen.setCosmetic(true);
	 QBrush selectedBrush (path->getColor());
	 GraphEditor_ListBezierPath *item = new GraphEditor_ListBezierPath(path->GetListeCtrls(),selectedPen, selectedBrush, itemMenu,this,path->isClosedPath());
	 item->setZValue(5000);
	// GraphEditor_PolyLineShape* polyline = dynamic_cast<GraphEditor_PolyLineShape*>(item);
	// connect(item,SIGNAL(polygonChanged(QVector<QPointF>,bool)),this,SLOT(onPolygonChanged(QVector<QPointF>,bool)));
//	connect(item,SIGNAL(polygonChanged(QVector<PointCtrl>, QVector<QPointF>,bool)),this, SLOT(onPolygonChangedTangent(QVector<PointCtrl>,QVector<QPointF>,bool)));
	connect(item,SIGNAL(polygonChanged2(GraphEditor_ListBezierPath*)),this, SLOT(onPolygonChangedTangent(GraphEditor_ListBezierPath*)));

	connect(this,SIGNAL(updateWidthRandom(int,int)),item,SLOT(polygonResize(int,int)));

	 addItem(item);

	 item->setSelected(true);*/



}


GraphEditor_CurveShape* GraphicSceneEditor::addNewCurve(QPolygonF poly, bool isopen)
{

	 QPen selectedPen(QColor(0,0,255,255), 3, Qt::DashLine);
	 selectedPen.setCosmetic(true);
	 QBrush selectedBrush (QColor(0,0,128,255));

	 GraphEditor_CurveShape *item = new GraphEditor_CurveShape(poly,eShape_Bezier_Curve,selectedPen, selectedBrush, itemMenu,this,!isopen);
	 item->setZValue(5000);
	// GraphEditor_PolyLineShape* polyline = dynamic_cast<GraphEditor_PolyLineShape*>(item);
	 connect(item,SIGNAL(polygonChanged(QVector<QPointF>,bool)),this,SLOT(onPolygonChanged(QVector<QPointF>,bool)));

	 addItem(item);

	 return item;
}

GraphEditor_ListBezierPath* GraphicSceneEditor::addNewCurve(QVector<PointCtrl> listepoints, bool isopen,QString name)
{

	 QPen selectedPen(QColor(0,0,255,255), 3, Qt::DashLine);
	 selectedPen.setCosmetic(true);
	 QBrush selectedBrush (QColor(0,0,128,255));

	// for(int i=0;i<listepoints.count();i++)
	//	 qDebug()<<i<<" , pos "<<listepoints[i]->m_position;

	 QColor col = NurbsWidget::getCurrentColor();

	 GraphEditor_ListBezierPath *item = new GraphEditor_ListBezierPath(listepoints,selectedPen, selectedBrush, itemMenu,this,!isopen,col);
	 item->setNameNurbs(name);
	 item->setZValue(5000);
	// GraphEditor_PolyLineShape* polyline = dynamic_cast<GraphEditor_PolyLineShape*>(item);
	// connect(item,SIGNAL(polygonChanged(QVector<QPointF>,bool)),this,SLOT(onPolygonChanged(QVector<QPointF>,bool)));
//	connect(item,SIGNAL(polygonChanged(QVector<PointCtrl>, QVector<QPointF>,bool)),this, SLOT(onPolygonChangedTangent(QVector<PointCtrl>,QVector<QPointF>,bool)));
	connect(item,SIGNAL(polygonChanged2(GraphEditor_ListBezierPath*)),this, SLOT(onPolygonChangedTangent(GraphEditor_ListBezierPath*)));

	connect(this,SIGNAL(updateWidthRandom(int,int)),item,SLOT(polygonResize(int,int)));

	 addItem(item);

	 return item;
}

GraphEditor_ListBezierPath* GraphicSceneEditor::addNewCurve(GraphEditor_ListBezierPath* path)
{

	 QPen selectedPen(QColor(0,0,255,255), 3, Qt::DashLine);
	 selectedPen.setCosmetic(true);
	 QBrush selectedBrush (QColor(0,0,128,255));


	// qDebug()<<"TODO !"<<path->GetListeCtrls().count();


	 return nullptr;
	 QVector<PointCtrl> listepoints = path->GetListeCtrls();


	 GraphEditor_ListBezierPath *item = new GraphEditor_ListBezierPath(listepoints,selectedPen, selectedBrush, itemMenu,this,path->isClosedPath());
	 item->setZValue(5000);

	connect(item,SIGNAL(polygonChanged2(GraphEditor_ListBezierPath*)),this, SLOT(onPolygonChangedTangent(GraphEditor_ListBezierPath*)));

	connect(this,SIGNAL(updateWidthRandom(int,int)),item,SLOT(polygonResize(int,int)));

	 addItem(item);

	 return item;
}

/*
void GraphicSceneEditor::receiveIndexCurrent(int index)
{
	qDebug()<<"index current : "<<index;
	emit sendIndexChanged(index);
}
*/
void GraphicSceneEditor::deleteItem(QGraphicsItem* item)
{
	if(item != nullptr)
	{
		removeItem(item);
		delete item;
	}
}
/************************************************************/
/* Public Slots */
/************************************************************/
void GraphicSceneEditor::GraphicToolNewAction(eGraphicAction action, st_GraphicToolsSettings st_GraphicSettings)
{
	//qDebug() << st_GraphicSettings.viewName  << m_InnerView->title();
	if (st_GraphicSettings.viewName == m_InnerView->title())
	{
		// If there is draw pending, just end it
		if (!m_EndPicking)
		{
			if (m_item)
				removeItem(m_item);
			m_EndPicking = 1;
		}

		if (action == eGraphicAction_Copy)
		{
			copyItem();
		}
		else if (action == eGraphicAction_Paste)
		{
			pasteItem();
		}
		else if (action == eGraphicAction_Cut)
		{
			cutItem();
		}
		else if (action == eGraphicAction_Delete)
		{
			deleteItem();
		}
		else if (st_GraphicSettings.action == eGraphicAction_Fill)
		{
			foreach(QGraphicsItem* p, selectedItems()) {
				//if (! dynamic_cast<GraphEditor_PolyLineShape*>(p) )
				{
					if (dynamic_cast<GraphEditor_LineShape *>(p))
						dynamic_cast<GraphEditor_LineShape*>(p)->setPen(st_GraphicSettings.pen);
					else if (dynamic_cast<QAbstractGraphicsShapeItem*>(p))
					{
						dynamic_cast<QAbstractGraphicsShapeItem*>(p)->setPen(st_GraphicSettings.pen);
						dynamic_cast<QAbstractGraphicsShapeItem*>(p)->setBrush(st_GraphicSettings.brush);
					}
				}
				//else
				//dynamic_cast<GraphEditor_PolyLineShape*>(p)->setPen(st_GraphicSettings.pen);
			}
		}
		else if (action == eGraphicAction_BringFront)
		{
			bringToFront();
		}
		else if (action == eGraphicAction_SendBack)
		{
			sendToBack();
		}
		else if (action == eGraphicAction_Undo)
		{
			undo();
		}
		else if (action == eGraphicAction_Redo)
		{
			redo();
		}
		else if (action == eGraphicAction_Erase)
		{
			deleteInsideData();
		}
	}
	//GraphicToolsWidget::setDefaultAction();

	//	if (st_GraphicSettings.action != eGraphicAction_Draw)
	//	{
	//		GraphicToolsWidget::setDefaultAction();
	//	}
}

void GraphicSceneEditor::updateSelectedItemsPen(QPen pen, ePenProperties property)
{
    if (!selectedItems().empty())
    {
        foreach(QGraphicsItem* p, selectedItems()) {
            if (dynamic_cast<QAbstractGraphicsShapeItem*>(p) || dynamic_cast<GraphEditor_LineShape*>(p))
            {
                QPen oldpen;
                if (dynamic_cast<GraphEditor_LineShape*>(p))
                    oldpen = dynamic_cast<GraphEditor_LineShape*>(p)->pen();
                else
                    oldpen = dynamic_cast<QAbstractGraphicsShapeItem*>(p)->pen();

                if (property == e_PenColor)
                {
                    oldpen.setColor(pen.color());
                }
                else if (property == e_PenWidth)
                {
                    oldpen.setWidth(pen.width());
                }
                else if (property == e_PenStyle)
                {
                    oldpen.setStyle(pen.style());
                }
                else if (property == e_PenJoinStyle)
                {
                    oldpen.setJoinStyle(pen.joinStyle());
                }
                else if (property == e_PenCap)
                {
                    oldpen.setCapStyle(pen.capStyle());
                }
                if (dynamic_cast<GraphEditor_LineShape*>(p))
                    dynamic_cast<GraphEditor_LineShape*>(p)->setPen(oldpen);
                else
                    dynamic_cast<QAbstractGraphicsShapeItem*>(p)->setPen(oldpen);
                backupUndostack();
            }
        }
    }
}

void GraphicSceneEditor::updateSelectedItemsBrush(QBrush brush, eBrushProperties property){
    if (!selectedItems().empty()){
        foreach(QGraphicsItem* p, selectedItems()) {
            //if (! dynamic_cast<GraphEditor_PolyLineShape*>(p) )
            {
                if (dynamic_cast<QAbstractGraphicsShapeItem*>(p)){
                    QBrush oldBrush = dynamic_cast<QAbstractGraphicsShapeItem*>(p)->brush();
                    if (property == e_BrushColor)
                    {
                        oldBrush.setColor(brush.color());
                    }
                    else if (property == e_BrushStyle)
                    {
                        oldBrush.setStyle(brush.style());
                        oldBrush.setColor(brush.color());
                    }
                    dynamic_cast<QAbstractGraphicsShapeItem*>(p)->setBrush(oldBrush);
                    backupUndostack();
                }
            }
        }
    }
}

void GraphicSceneEditor::updateSelectedTextColor(QColor clr)
{
    foreach(QGraphicsItem* p, selectedItems()) {
        if (dynamic_cast<GraphEditor_TextItem*>(p))
        {
            if (dynamic_cast<GraphEditor_TextItem*>(p)->defaultTextColor() != clr)
            {
                dynamic_cast<GraphEditor_TextItem*>(p)->setDefaultTextColor(clr);
                backupUndostack();
            }
        }
    }
}

void GraphicSceneEditor::updateSelectedTextFont(QFont font)
{
    foreach(QGraphicsItem* p, selectedItems()) {
        if (dynamic_cast<GraphEditor_TextItem*>(p))
        {
            if (dynamic_cast<GraphEditor_TextItem*>(p)->font() != font)
            {
                dynamic_cast<GraphEditor_TextItem*>(p)->setFont(font);
                backupUndostack();
            }
        }
    }
}

void GraphicSceneEditor::updateSelectedCurveSmooth(int newSmoothValue){
    if (!selectedItems().empty()){
        foreach(QGraphicsItem* p, selectedItems()) {
            if ( dynamic_cast<GraphEditor_RegularBezierPath*>(p) ){
                dynamic_cast<GraphEditor_RegularBezierPath*>(p)->setSmoothValue(newSmoothValue);


                backupUndostack();
            }
        }
    }
}

/************************************************************/
/* Private Functions */
/************************************************************/

void GraphicSceneEditor::CreateRandomView(){
	QList<QGraphicsItem*> selectedItemsList = selectedItems();
	if (selectedItemsList.count()==0)
	{
		return;
	}

    GraphEditor_ItemInfo* pItem = dynamic_cast<GraphEditor_ItemInfo*> (selectedItemsList.first());
    if (pItem){
        if (m_InnerView->geotimeView()){
            MultiTypeGraphicsView* pGView = dynamic_cast<MultiTypeGraphicsView*>(m_InnerView->geotimeView());
            if(pGView != nullptr){
                RandomLineView *pRandomView = pGView->createRandomView(pItem->SceneCordinatesPoints());
                pItem->setRandomView(pRandomView);
                pRandomView->setItemSection(pItem);

            }

        }
        if(m_InnerView)
		{

			m_InnerView->showRandomView(false,pItem->SceneCordinatesPoints());
		}
    }
}

void GraphicSceneEditor::deleteOrthoItem(GraphEditor_LineShape* pOrthItem){

    if(pOrthItem!=nullptr){

    	bool exist= m_GraphicItems[m_SliceValue].contains(pOrthItem);
    	if(exist)
    	{
			disconnect(pOrthItem,nullptr,nullptr,nullptr);
			m_GraphicItems[m_SliceValue].removeAll(pOrthItem);
			removeItem(pOrthItem);
			delete pOrthItem;
			pOrthItem = nullptr;
    	}

        //deleteItem(pOrthItem);//removeItem(pOrthItem);
       // delete pOrthItem;
    }
}



bool  GraphicSceneEditor::createOrgthognalRandomView(QString name, QColor col)
{

	 if (selectedItems().isEmpty())
	       return false;

   GraphEditor_ItemInfo* pItem = dynamic_cast<GraphEditor_ItemInfo*> (selectedItems().first()); //dynamic_cast<GraphEditor_ItemInfo*>(getCurrentBezier(name)) ;//dynamic_cast<GraphEditor_ItemInfo*> (selectedItems().first());
    if (pItem){
    	// std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    	nurbs3d(name,col);
        if (m_InnerView->geotimeView()){
            QLineF line;
            QVector<QPointF> vectPoint = pItem->SceneCordinatesPoints();
            for(int i = 0;i < vectPoint.size();i++){
                if(i <= vectPoint.size()- 2)
                {
                    if(vectPoint[i] != vectPoint[i+1]){
                        line = QLineF(vectPoint[i],vectPoint[i+1]).normalVector();
                        //qDebug()<<" ==> line : "<<line;
                        break;
                    }
                }
            }
            line.setLength(1500);
            QLineF line2 = line;
            line2.setAngle(line.angle()+180);

            /*! create orthogonal line */
            GraphEditor_LineShape *pOrthoLine = new GraphEditor_LineShape(QLineF(line.p2(),line2.p2()),GraphicToolsWidget::getPaletteSettings().pen,itemMenu,this);


            if(pOrthoLine != nullptr){
                //GraphEditor_ItemInfo * pOrthoItem = dynamic_cast<GraphEditor_ItemInfo*>(pOrthoLine);
                pOrthoLine->setZValue(99);
                pOrthoLine->setSelected(true);
                pOrthoLine->setFlag(QGraphicsItem::ItemSendsGeometryChanges,true);
                pOrthoLine->setFlag(QGraphicsItem::ItemIsMovable,true);
                addItem(pOrthoLine);
                m_GraphicItems[m_SliceValue].append(pOrthoLine);
               // SceneStateMirror(pOrthoLine);


                pOrthoLine->setOrthogonal(pItem);

                RandomLineView *pRandomView = pItem->getRandomView();
                RandomLineView *pOrthRandomView = nullptr;
                MultiTypeGraphicsView* pGView = dynamic_cast<MultiTypeGraphicsView*>(m_InnerView->geotimeView());
                if(pRandomView == nullptr){
                    if(pGView != nullptr){
                         pRandomView = pGView->createRandomView(pItem->SceneCordinatesPoints());
                    }
                }
                if(pGView != nullptr){
                    pOrthRandomView = pGView->createRandomView(pOrthoLine->SceneCordinatesPoints(),eTypeOrthogonal);

                    pOrthRandomView->setCurrentNaneNurbs(name);
                    pOrthRandomView->setDefaultTitle("Random-"+name);
                    if(pRandomView != nullptr){

                        pRandomView->setOrthogonalView(pOrthRandomView);
                        if(m_bezierSelected !=nullptr) m_bezierSelected->setNameNurbs(name);
                        pOrthRandomView->m_BezierDirectrice =  m_bezierSelected;
                        RandomRep *slice = pOrthRandomView->firstRandom();

                        NurbsWidget::setInlineView(pOrthRandomView);

                    if(slice != nullptr)
                    {
						QPolygon discreate;
						QPolygonF worldDiscreate;
						RandomLineView::discreatePoly(pItem->SceneCordinatesPoints(),dynamic_cast<Seismic3DAbstractDataset*>(slice->data()),discreate ,worldDiscreate);
						int width = discreate.size();
						pOrthRandomView->defineSliceMinMax(0, width);//slice->image());
						pOrthRandomView->setOrthoWorldDiscreatePolyline(worldDiscreate);
						pOrthRandomView->setOrthoDiscreatePolyline(discreate);




                    }
                        pOrthRandomView->addOrthogonalLine(pOrthoLine);
                        pOrthRandomView->setColorCross(col);
                        pOrthoLine->setRandomView(pOrthRandomView);
                        connect(pOrthRandomView,SIGNAL(newPointPosition(QPointF)),pOrthoLine, SLOT(updateOrtholine(QPointF)));
                        connect(pOrthRandomView,SIGNAL(newWidthOrthogonal(double)),pOrthoLine,SLOT(updateOrthoWidthline(double)));
                        connect(pOrthoLine,SIGNAL(orthogonalUpdated(QPolygonF)),pOrthRandomView,SLOT(setPolyLine(QPolygonF)));


                        connect(pOrthRandomView,SIGNAL(signalMoveCrossFinish()),m_bezierSelected, SLOT(polygonChanged1()));
                        connect(pOrthRandomView,SIGNAL(signalAddPointsDirectrice(QPointF)),m_bezierSelected, SLOT(receiveAddPts(QPointF)));
                        connect(pOrthRandomView,SIGNAL(signalMovePointsDirectrice(int,int,int)),m_bezierSelected, SLOT(moveGrabber(int,int,int)));
                        connect(pOrthRandomView,SIGNAL(signalPositionPointsDirectrice(int,QPointF)),m_bezierSelected, SLOT(positionGrabber(int,QPointF)));
                        connect(m_bezierSelected,SIGNAL(signalCurrentIndex(int)),pOrthRandomView, SLOT(receiveCurrentGrabberIndex(int)));

                        GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(findPath());
						if(listBezier != nullptr)
						{
							//connect(pOrthRandomView, SIGNAL(updateOrthoFrom3D(QVector3D,QPointF )),pOrthoLine,SLOT(refreshOrtholine(QVector3D,QPointF)));
						}
						else
						{
							connect(pOrthRandomView, SIGNAL(updateOrthoFrom3D(QVector3D,QPointF )),pOrthoLine,SLOT(refreshOrtholine(QVector3D,QPointF)));
						}


                        connect(pOrthRandomView,SIGNAL(linedeteled(GraphEditor_LineShape*)),this,SLOT(deleteOrthoItem(GraphEditor_LineShape*)));




                        if(m_InnerView)
						{


							//m_InnerView->showRandomView(true,pOrthoLine->SceneCordinatesPoints());
							m_InnerView->showRandomView(true,pOrthoLine,pOrthRandomView,name);
							// qDebug()<<" create ortho line ....3";
							  //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
							 // qDebug() <<" createOrgthognalRandomView , Time : " << std::chrono::duration<double, std::milli>(end-start).count();
						}
                    }
                }
            }
        }
        return true;
    }
    else
    {
    	qDebug()<<" Bezier curve is not selected !!";
    	return false;
    }
}

void GraphicSceneEditor::bringToFront(){
    if (selectedItems().isEmpty())
        return;

    QGraphicsItem *selectedItem = selectedItems().first();
    QList<QGraphicsItem *> overlapItems = selectedItem->collidingItems();

    qreal zValue = 0;
    foreach (QGraphicsItem *item, overlapItems) {
        if (item->zValue() >= zValue)
        {
            zValue = item->zValue() + 1;
        }
    }
    selectedItem->setZValue(zValue);
    backupUndostack();
}

void GraphicSceneEditor::sendToBack()
{
    if (selectedItems().isEmpty())
        return;

    QGraphicsItem *selectedItem = selectedItems().first();
    QList<QGraphicsItem *> overlapItems = selectedItem->collidingItems();

    qreal zValue = selectedItem->zValue();
    foreach (QGraphicsItem *item, overlapItems) {
        /* Shape items zValues > 4000
         * biggest Z value is = Abstract2DInnerView::CROSS_ITEM_Z = 3000; */
        if ((item->zValue() > 3000) && (item->zValue() <= zValue))
        {
            zValue = item->zValue() - 1;
        }
    }
    selectedItem->setZValue(zValue);
    backupUndostack();
}

void GraphicSceneEditor::deleteItems(QList<QGraphicsItem*> const& items)
{
    foreach (QGraphicsItem *item, items) {
        if ((item) && (item->scene() != nullptr))
        {
            removeItem(item);
            delete item;
            item=nullptr;
        }
    }
}

void GraphicSceneEditor::copyItem()
{
    foreach(QGraphicsItem* p, m_PasteBoard) {
        delete p;
    }
    m_PasteBoard = cloneItems(selectedItems());
}

void GraphicSceneEditor::pasteItem()
{
    st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();
    QList<QGraphicsItem*> m_PasteBoardCopy(cloneItems(m_PasteBoard));
    foreach(QGraphicsItem* p, items()) p->setSelected(false);

    foreach(QGraphicsItem* item, m_PasteBoard) {
        item->setPos(item->scenePos() + QPointF(20, 20));
        addItem(item);
        //item->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
        item->setSelected(true);
        qreal maxZ = 0;
        foreach (QGraphicsItem *p,
                item->collidingItems(Qt::IntersectsItemBoundingRect))
        maxZ = qMax(maxZ, item->zValue());
        item->setZValue(maxZ);
        m_GraphicItems[m_SliceValue].append(item);
    }
    if (!m_PasteBoard.empty())
        backupUndostack();

    m_PasteBoard.swap(m_PasteBoardCopy);

}

void GraphicSceneEditor::cutItem()
{
    copyItem();
    deleteItem();
    backupUndostack();
}

void GraphicSceneEditor::deleteMyItem(int slice, QGraphicsItem* item)
{
	foreach(QGraphicsItem *p, m_GraphicItems[slice])
		{
		if (p==item)
		{
			 m_GraphicItems[m_SliceValue].removeAll(p);
			 removeItem(p);
			 delete p;
			 p = nullptr;
		}
	}

}

void GraphicSceneEditor::deleteItem()
{
    bool needsBackup = !selectedItems().empty();
    foreach(QGraphicsItem *p, m_GraphicItems[m_SliceValue]) {
        if (p)
        {
            if (p->isSelected())
            {
                m_GraphicItems[m_SliceValue].removeAll(p);

                GraphEditor_LineShape* shape = (dynamic_cast<GraphEditor_LineShape *>(p));
				if(shape != nullptr)
				{
					RandomLineView* view  = shape->getRandomView();
					if( view != nullptr)
					{
						view->deleteCurves();
						if(m_InnerView != nullptr) m_InnerView->randomLineDeleted(view);
					}
				}

				GraphEditor_ItemInfo* pItem = dynamic_cast<GraphEditor_ItemInfo*>(p);
				if(pItem != nullptr)
				{

					RandomLineView* view  = pItem->getRandomView();
					if( view != nullptr)
					{
						if(m_InnerView != nullptr) m_InnerView->randomLineDeleted(view);
					}
				}

                removeItem(p);
                delete p;
            }
        }
        else
        {
            m_GraphicItems[m_SliceValue].removeAll(p);
        }
    }
    if (needsBackup)
        backupUndostack();
}

void GraphicSceneEditor::rotate(signed short degree)
{
    foreach(QGraphicsItem *p, selectedItems() )    {
        QString txtRot = p->data(0).toString();
        if ( txtRot.compare("noRotation") == 0 ) continue;
        QRectF bbox = p->boundingRect().normalized();
        QPointF center = bbox.center();
        p->setTransformOriginPoint(center);
        p->setRotation(p->rotation()+ degree);
    }
    backupUndostack();
}

void GraphicSceneEditor::undo()
{
    if (undoStack.isEmpty())
    {
        //qDebug() << "undo stack empty";
        return;
    }
    // sweep away all items
    deleteItems(m_GraphicItems[m_SliceValue]);
    m_GraphicItems[m_SliceValue].clear();
    QList<QGraphicsItem*> undoneItems = cloneItems(undoStack.undo());
    m_GraphicItems[m_SliceValue]=undoneItems;
    foreach(QGraphicsItem* item, undoneItems) {
        addItem(item);
        //item->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
        item->setSelected(false);
    }
    SceneStateMirror();
}

void GraphicSceneEditor::redo()
{
    if (undoStack.isFull()) return;
    deleteItems(m_GraphicItems[m_SliceValue]);
    m_GraphicItems[m_SliceValue].clear();
    QList<QGraphicsItem*> redoneItems = cloneItems(undoStack.redo());
    m_GraphicItems[m_SliceValue]=redoneItems;
    foreach(QGraphicsItem* item, redoneItems) {
        addItem(item);
        //item->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
        item->setSelected(false);
    }
    SceneStateMirror();
}

void GraphicSceneEditor::groupItems()
{
    QGraphicsItemGroup* group = createItemGroup(selectedItems());
    group->setFlag(QGraphicsItem::ItemIsMovable, true);
    group->setFlag(QGraphicsItem::ItemIsSelectable, true);
    group->setFlag(QGraphicsItem::ItemSendsGeometryChanges, true);
    addItem(group);
    backupUndostack();
}

void GraphicSceneEditor::ungroupItems()
{
    foreach(QGraphicsItem* p, selectedItems()) {
        if (p->type() == QGraphicsItemGroup::Type)
        {
            destroyItemGroup(qgraphicsitem_cast<QGraphicsItemGroup*>(p));
        }
    }
    backupUndostack();
}

void GraphicSceneEditor::SceneStateMirror()
{
    if (((m_InnerView->viewType() == BasemapView ) || (m_InnerView->viewType() == StackBasemapView ))&& (m_InnerView->geotimeView() != nullptr)){

        GeotimeGraphicsView* pGView = dynamic_cast<GeotimeGraphicsView*>(m_InnerView->geotimeView());
        if(pGView != nullptr){
            foreach (AbstractInnerView *p , pGView->getInnerViews()){
                if (p == m_InnerView)
                {
                    continue;
                }
                if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
                {
                    QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
                    if (scene)
                    {
                        dynamic_cast<GraphicSceneEditor *>(scene)->saveItemFromOtherScene(m_GraphicItems[m_SliceValue],undoStack);

                    }
                }
            }
        }else {
            SplittedView* pGView = dynamic_cast<SplittedView*>(m_InnerView->geotimeView());
            if(pGView != nullptr){
                foreach (AbstractInnerView *p , pGView->getInnerViews()){
                    if (p == m_InnerView)
                    {
                        continue;
                    }
                    if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
                    {
                        QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
                        if (scene)
                        {
                            dynamic_cast<GraphicSceneEditor *>(scene)->saveItemFromOtherScene(m_GraphicItems[m_SliceValue],undoStack);
                        }
                    }
                }
            }
        }
    }
}

void GraphicSceneEditor::SceneStateMirror(QGraphicsItem * item)
{
    if (((m_InnerView->viewType() == BasemapView ) || (m_InnerView->viewType() == StackBasemapView ))&& (m_InnerView->geotimeView() != nullptr)){

        GeotimeGraphicsView* pGView = dynamic_cast<GeotimeGraphicsView*>(m_InnerView->geotimeView());
        if(pGView != nullptr){
            foreach (AbstractInnerView *p , pGView->getInnerViews()){
                if (p == m_InnerView)
                {
                    continue;
                }
                if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
                {
                    QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
                    if (scene)
                    {
                        dynamic_cast<GraphicSceneEditor *>(scene)->addItem(item);
                        dynamic_cast<GraphicSceneEditor *>(scene)->m_GraphicItems[m_SliceValue].append(item);
                    }
                }
            }
        }else {
            SplittedView* pGView = dynamic_cast<SplittedView*>(m_InnerView->geotimeView());
            if(pGView != nullptr){
                foreach (AbstractInnerView *p , pGView->getInnerViews()){
                    if (p == m_InnerView)
                    {
                        continue;
                    }
                    if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
                    {
                        QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
                        if (scene)
                        {
                            dynamic_cast<GraphicSceneEditor *>(scene)->addItem(item);
                            dynamic_cast<GraphicSceneEditor *>(scene)->m_GraphicItems[m_SliceValue].append(item);
                        }
                    }
                }
            }
        }
    }
}



void GraphicSceneEditor::backupUndostack()
{
    undoStack.backup(cloneItems(m_GraphicItems[m_SliceValue]));
    SceneStateMirror();
}

QRectF GraphicSceneEditor::CreateRect(QPointF p1, QPointF p2)
{
    return QRectF(QPointF(qMin(p1.x(),p2.x()), qMin(p1.y(),p2.y())), QPointF(qMax(p1.x(),p2.x()), qMax(p1.y(),p2.y())));
}

void GraphicSceneEditor::saveItem(){
    m_EndPicking =1;
    if (!m_LayersMap.count(m_SliceValue)){
        WorkingSetManager* workingmanager = GeotimeGraphicsView::getWorkingSetManager();
        GraphicTool_GraphicLayer* layer = new GraphicTool_GraphicLayer(workingmanager,QString::number(m_SliceValue),QList<QGraphicsItem *>(),eSliceLayer,m_InnerView);
        workingmanager->addGraphicLayer(layer);
        layer->setAllDisplayPreference(true);
        m_LayersMap[m_SliceValue]=layer;
    }
    m_GraphicItems[m_SliceValue].append(m_item);
    backupUndostack();
    GraphicToolsWidget::setDefaultAction();
}

void GraphicSceneEditor::saveItem(QGraphicsItem *pItem){
    m_EndPicking =1;
    if (!m_LayersMap.count(m_SliceValue)){
        WorkingSetManager* workingmanager = GeotimeGraphicsView::getWorkingSetManager();
        GraphicTool_GraphicLayer* layer = new GraphicTool_GraphicLayer(workingmanager,QString::number(m_SliceValue),QList<QGraphicsItem *>(),eSliceLayer,m_InnerView);
        workingmanager->addGraphicLayer(layer);
        layer->setAllDisplayPreference(true);
        m_LayersMap[m_SliceValue]=layer;
    }
    m_GraphicItems[m_SliceValue].append(pItem);
    backupUndostack();
    GraphicToolsWidget::setDefaultAction();
}

void GraphicSceneEditor::saveItemFromOtherScene(QList<QGraphicsItem*> items, UndoSystem undosystem)
{
    //if (m_GraphicItems[m_SliceValue].empty())
    if (!m_LayersMap.count(m_SliceValue))
    {
        WorkingSetManager* workingmanager = GeotimeGraphicsView::getWorkingSetManager();
		GraphicTool_GraphicLayer *layer = new GraphicTool_GraphicLayer(
				workingmanager, QString::number(m_SliceValue),
				QList<QGraphicsItem*>(), eSliceLayer, m_InnerView);
        workingmanager->addGraphicLayer(layer);
        layer->setAllDisplayPreference(true);
        m_LayersMap[m_SliceValue]=layer;
    }

    deleteItems(m_GraphicItems[m_SliceValue]);
    m_GraphicItems[m_SliceValue].clear();
    m_GraphicItems[m_SliceValue]=cloneItems(items);
    foreach(QGraphicsItem* item, m_GraphicItems[m_SliceValue]) {
        addItem(item);
        //item->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
        item->setSelected(false);
        //if (!m_LayersMap[m_SliceValue]->displayPreference())
        //{
        //    item->hide();
        //}
//        GraphEditor_ItemInfo* pItem = dynamic_cast<GraphEditor_ItemInfo*>(item);
//               if(pItem->getRandomView() != nullptr){
//                   if(pItem->getRandomView()->getRandomType() == eTypeOrthogonal){
//
//                   }
//               }
    }
    undoStack = undosystem;
}


QList<QGraphicsItem*> GraphicSceneEditor::cloneItems(const QList<QGraphicsItem*>& items){
    QHash<QGraphicsItem*, QGraphicsItem*> copyMap;


    foreach (QGraphicsItem* item, items) {

        if (dynamic_cast<GraphEditor_LineShape*>(item))
            copyMap[item] = qgraphicsitem_cast<GraphEditor_LineShape*>(item)->clone();
        else if (dynamic_cast<GraphEditor_RectShape*>(item))
            copyMap[item] = qgraphicsitem_cast<GraphEditor_RectShape*>(item)->clone();
        else if (dynamic_cast<GraphEditor_EllipseShape*>(item))
            copyMap[item] = qgraphicsitem_cast<GraphEditor_EllipseShape*>(item)->clone();
        else if (dynamic_cast<GraphEditor_PolygonShape*>(item))
            copyMap[item] = qgraphicsitem_cast<GraphEditor_PolygonShape*>(item)->clone();
        else if (dynamic_cast<GraphEditor_RegularBezierPath*>(item))
        {
            copyMap[item] = qgraphicsitem_cast<GraphEditor_RegularBezierPath*>(item)->clone();
            m_bezierSelected =  dynamic_cast<GraphEditor_Path*>(copyMap[item]);
            connect(m_bezierSelected,SIGNAL(polygonChanged(QVector<QPointF>,bool)),this, SLOT(onPolygonChanged(QVector<QPointF>,bool)));
        }
        else if (dynamic_cast<GraphEditor_ListBezierPath*>(item))
		   {
			   copyMap[item] = qgraphicsitem_cast<GraphEditor_ListBezierPath*>(item)->clone();
				//GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(copyMap[item]);
			 //  qDebug()<<"BEZIER CLONE ITEM";
			  // connect(copyMap[item],SIGNAL(polygonChanged(QVector<QPointF>,bool)),this, SLOT(onPolygonChanged(QVector<QPointF>,bool)));
			   //m_bezierSelected =  dynamic_cast<GraphEditor_Path*>(copyMap[item]);
			  //connect(listBezier,SIGNAL(polygonChanged(QVector<PointCtrl>, QVector<QPointF>,bool)),this, SLOT(onPolygonChangedTangent(QVector<PointCtrl>,QVector<QPointF>,bool)));

		   }
        else if (dynamic_cast<GraphEditor_CurveShape*>(item))
            copyMap[item] = qgraphicsitem_cast<GraphEditor_CurveShape*>(item)->clone();
        else if (dynamic_cast<GraphEditor_PolyLineShape*>(item))
        {
            copyMap[item] = qgraphicsitem_cast<GraphEditor_PolyLineShape*>(item)->clone();
            m_bezierSelected =  dynamic_cast<GraphEditor_Path*>(copyMap[item]);
            connect(m_bezierSelected,SIGNAL(polygonChanged(QVector<QPointF>,bool)),this, SLOT(onPolygonChanged(QVector<QPointF>,bool)));

        }
        else if (dynamic_cast<GraphEditor_Path*>(item))
            copyMap[item] = qgraphicsitem_cast<GraphEditor_Path*>(item)->clone();
        else if (dynamic_cast<GraphEditor_TextItem*>(item))
            copyMap[item] = qgraphicsitem_cast<GraphEditor_TextItem*>(item)->clone();
    }
    return copyMap.values();
}

void GraphicSceneEditor::deleteInsideData()
{
    foreach(QGraphicsItem* p, selectedItems()) m_InnerView->deleteData(p);
}

void GraphicSceneEditor::deselectWells()
{
    foreach(QGraphicsItem* p, selectedItems()) m_InnerView->deselectWellsIncludedInItem(p);
}

QVariant GraphicSceneEditor::item_to_variant(QGraphicsItem* item)
{
    QVariantHash data;
    //save all needed attributes
    if (dynamic_cast<GraphEditor_LineShape *>(item))
    {
        data["form"] = "line";
        data["line"] = dynamic_cast<GraphEditor_LineShape *>(item)->line();
    }
    else if (dynamic_cast<GraphEditor_RectShape *>(item))
    {
        data["form"] = "rect";
        data["rect"] = dynamic_cast<GraphEditor_RectShape *>(item)->rect();
    }
    else if (dynamic_cast<GraphEditor_EllipseShape *>(item))
    {
        data["form"] = "ellipse";
        data["rect"] = dynamic_cast<GraphEditor_EllipseShape *>(item)->rect();
    }
    else if (dynamic_cast<GraphEditor_CurveShape *>(item))
    {
        data["form"] = "curve";
        data["polygon"] = dynamic_cast<GraphEditor_CurveShape *>(item)->polygon();
        data["type"] = dynamic_cast<GraphEditor_CurveShape *>(item)->type();
        data["isClosedPath"]  = dynamic_cast<GraphEditor_CurveShape *>(item)->isClosedPath();
    }
    else if (dynamic_cast<GraphEditor_PolyLineShape *>(item))
    {
        data["form"] = "polyline";
        data["polygon"] = dynamic_cast<GraphEditor_PolyLineShape *>(item)->polygon();
        data["isClosedPath"]  = dynamic_cast<GraphEditor_PolyLineShape *>(item)->isClosedPath();
    }
    else if (dynamic_cast<GraphEditor_ListBezierPath *>(item))
	{
		data["form"] = "listbezierpath";
	//	data["initialPolygon"] = dynamic_cast<GraphEditor_ListBezierPath *>(item)->initialPolygon();
		data["polygon"] = dynamic_cast<GraphEditor_ListBezierPath *>(item)->polygon();
		data["isClosedCurve"] = dynamic_cast<GraphEditor_ListBezierPath *>(item)->isClosedPath() ;//isClosedCurve();

	}

    else if (dynamic_cast<GraphEditor_RegularBezierPath *>(item))
    {
        data["form"] = "bezierpath";
        data["initialPolygon"] = dynamic_cast<GraphEditor_RegularBezierPath *>(item)->initialPolygon();
        data["polygon"] = dynamic_cast<GraphEditor_RegularBezierPath *>(item)->polygon();
        data["smooth"] = dynamic_cast<GraphEditor_RegularBezierPath *>(item)->smoothValue();
        data["isSmoothed"] = dynamic_cast<GraphEditor_RegularBezierPath *>(item)->isSmoothed();
        data["isClosedCurve"] = dynamic_cast<GraphEditor_RegularBezierPath *>(item)->isClosedPath() ;//isClosedCurve();
        //        qDebug() << "save isSmoothed " << data["isSmoothed"].toBool();
        //        qDebug() << "save isclosed " << data["isClosedCurve"].toBool();
        //        qDebug() << "save smoth =" << data["smooth"].toInt();
    }
    else if (dynamic_cast<GraphEditor_Path *>(item))
    {
        data["form"] = "path";
        data["polygon"] = dynamic_cast<GraphEditor_Path *>(item)->polygon();
        data["isClosedPath"]  = dynamic_cast<GraphEditor_Path *>(item)->isClosedPath();
    }
    else if (dynamic_cast<GraphEditor_PolygonShape *>(item))
    {
        data["form"] = "polygon";
        data["polygon"] = dynamic_cast<GraphEditor_PolygonShape *>(item)->polygon();
    }
    else if (dynamic_cast<GraphEditor_TextItem *>(item))
    {
        data["form"] = "text";
        data["plainText"] = dynamic_cast<GraphEditor_TextItem *>(item)->toPlainText();
        data["font"] = dynamic_cast<GraphEditor_TextItem *>(item)->font();
        data["textWidth"] = dynamic_cast<GraphEditor_TextItem *>(item)->textWidth();
        data["textColor"] = dynamic_cast<GraphEditor_TextItem *>(item)->defaultTextColor();
    }
    data["pos"] = item->scenePos();
    data["rotation"] = item->rotation();
    data["zvalue"] = item->zValue();
    if (dynamic_cast<QAbstractGraphicsShapeItem *>(item))
    {
        data["pen"] = dynamic_cast<QAbstractGraphicsShapeItem *>(item)->pen();
        data["brush"] = dynamic_cast<QAbstractGraphicsShapeItem *>(item)->brush();
    }
    else if (dynamic_cast<GraphEditor_LineShape *>(item))
    {
        data["pen"] = dynamic_cast<GraphEditor_LineShape *>(item)->pen();
    }
    return data;
}

QGraphicsItem* GraphicSceneEditor::item_from_variant(QVariant v)
{
    QVariantHash data = v.toHash();
    QGraphicsItem *result = nullptr;
    QPen loadedPen = data["pen"].value<QPen>();
    loadedPen.setCosmetic(true);
    QBrush loadedBrush = data["brush"].value<QBrush>();
    if (data["form"].toString() == "line")
    {
        result = new GraphEditor_LineShape(data["line"].toLineF(),loadedPen,itemMenu,this);
        dynamic_cast<GraphEditor_LineShape *>(result)->setLine(data["line"].toLineF());
    }
    else if (data["form"].toString() == "rect")
    {
        result = new GraphEditor_RectShape(data["rect"].toRectF(),loadedPen,loadedBrush,itemMenu);
        dynamic_cast<GraphEditor_RectShape *>(result)->setRect(data["rect"].toRectF());
    }
    else if (data["form"].toString() == "ellipse")
    {
        result = new GraphEditor_EllipseShape(data["rect"].toRectF(),loadedPen,loadedBrush,itemMenu);
        dynamic_cast<GraphEditor_EllipseShape *>(result)->setRect(data["rect"].toRectF());
    }
    else if (data["form"].toString() == "curve")
    {
        result = new GraphEditor_CurveShape(data["polygon"].value<QPolygonF>(),(eShape)data["type"].toInt(), loadedPen,loadedBrush,itemMenu, this,data["isClosedPath"].toBool());
    }
    else if (data["form"].toString() == "polyline")
    {
        result = new GraphEditor_PolyLineShape(data["polygon"].value<QPolygonF>(),loadedPen,loadedBrush,itemMenu,this,data["isClosedPath"].toBool());
    }
    else if (data["form"].toString() == "listbezierpath")
	{
		result = new GraphEditor_ListBezierPath(data["polygon"].value<QPolygonF>(),loadedPen,loadedBrush,
				itemMenu);
		dynamic_cast<GraphEditor_ListBezierPath *>(result)->restoreState(data["polygon"].value<QPolygonF>()	, data["isClosedCurve"].toBool());
	}
    else if (data["form"].toString() == "bezierpath")
    {
        result = new GraphEditor_RegularBezierPath(data["initialPolygon"].value<QPolygonF>(),loadedPen,loadedBrush,
                itemMenu);
        dynamic_cast<GraphEditor_RegularBezierPath *>(result)->restoreState(data["polygon"].value<QPolygonF>(),
                data["isSmoothed"].toBool(), data["isClosedCurve"].toBool(),data["smooth"].toInt());
    }
    else if (data["form"].toString() == "path")
    {
        result = new GraphEditor_Path(data["polygon"].value<QPolygonF>(),loadedPen,loadedBrush, itemMenu,data["isClosedPath"].toBool());
    }
    else if (data["form"].toString() == "polygon")
    {
        result = new GraphEditor_PolygonShape(data["polygon"].value<QPolygonF>(),loadedPen,loadedBrush,itemMenu);
    }
    else if (data["form"].toString() == "text")
    {
        result = new GraphEditor_TextItem();
        dynamic_cast<GraphEditor_TextItem *>(result)->setPlainText(data["plainText"].toString());
        dynamic_cast<GraphEditor_TextItem *>(result)->setFont(data["font"].value<QFont>());
        dynamic_cast<GraphEditor_TextItem *>(result)->setTextWidth(data["textWidth"].toReal());
        dynamic_cast<GraphEditor_TextItem *>(result)->setDefaultTextColor(data["textColor"].value<QColor>());
    }
    if (result)
    {
        result->setPos(data["pos"].toPointF());
        result->setRotation(data["rotation"].toDouble());
        result->setZValue(data["zvalue"].toDouble());
    }
    return result;
}

void GraphicSceneEditor::save_state(QString culturalName)
{
    QVariantList data_list;
    foreach(QGraphicsItem* item, m_GraphicItems[m_SliceValue]) {
        if (item->isVisible())
            data_list << item_to_variant(item);
    }

    QSettings settings(culturalName, QSettings::NativeFormat);
    settings.setValue("items", data_list);
    if ((m_InnerView->viewType() == InlineView) || (m_InnerView->viewType() == XLineView) )
    {
        if (dynamic_cast<SingleSectionView *> (m_InnerView))
        {
            settings.setValue("slice_value", dynamic_cast<SingleSectionView *> (m_InnerView)->sliceValueWorld());
        }
    }
}

void GraphicSceneEditor::addLayer(QString culturalName, QList<QGraphicsItem *> loadedItemsList)
{
    QFileInfo layerFile(culturalName);
    WorkingSetManager* workingmanager = GeotimeGraphicsView::getWorkingSetManager();
    GraphicTool_GraphicLayer* layer = new GraphicTool_GraphicLayer(workingmanager,layerFile.baseName(),loadedItemsList, eLoadedLayer, m_InnerView);
    workingmanager->addGraphicLayer(layer);
    layer->setAllDisplayPreference(true);
    backupUndostack();
}

void GraphicSceneEditor::restore_state(QString culturalName)
{
	st_GraphicToolsSettings st_GraphicSettings = GraphicToolsWidget::getPaletteSettings();
	if (!st_GraphicSettings.enabled)
	{
		GraphicToolsWidget::showPalette(m_InnerView->title());
	}
	QSettings settings(culturalName, QSettings::NativeFormat);
	if ((m_InnerView->viewType() == InlineView) || (m_InnerView->viewType() == XLineView) )
	{
		if (dynamic_cast<SingleSectionView *> (m_InnerView))
		{
			int slice_value = settings.value("slice_value").toInt();
			if (slice_value !=  dynamic_cast<SingleSectionView*> (m_InnerView)->sliceValueWorld())
			{
				int ret = QMessageBox::warning(m_InnerView, tr("Load Graphic layer"),
						tr("The selected graphic Layer belong to a different slice value.\n"
								"Continue loading?"),QMessageBox::Cancel,
								QMessageBox::Ok);
				if (ret != QMessageBox::Ok)
					return;
				else
					dynamic_cast<SingleSectionView*> (m_InnerView)->setSliceValue(slice_value);
			}
		}
	}
	QList<QGraphicsItem *> loadedItemsList;

	foreach(QVariant data, settings.value("items").toList())
	{
		QGraphicsItem* item = item_from_variant(data);
		if (item==nullptr) {
			continue;
		}
		item->setSelected(false);
		QFileInfo layerFile(culturalName);
		if(dynamic_cast<GraphEditor_Item *>(item))
		{
			dynamic_cast<GraphEditor_Item *>(item)->setGrabbersVisibility(false);
			dynamic_cast<GraphEditor_Item *>(item)->setID(layerFile.baseName());
		}
		addItem(item);
		//item->setCacheMode(QGraphicsItem::DeviceCoordinateCache);
		m_GraphicItems[m_SliceValue] << item;
		loadedItemsList << item;
	}
	addLayer(culturalName,loadedItemsList);
	if ((m_InnerView->viewType() == BasemapView ) || (m_InnerView->viewType() == StackBasemapView ) && (m_InnerView->geotimeView()!= nullptr))
	{
		GeotimeGraphicsView* pGView = dynamic_cast<GeotimeGraphicsView*>(m_InnerView->geotimeView());
		if(pGView != nullptr){
			foreach (AbstractInnerView *p , pGView->getInnerViews()) {
				if (p == m_InnerView)
				{
					continue;
				}
				if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
				{
					QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
					if (scene)
					{
						dynamic_cast<GraphicSceneEditor *>(scene)->addLayer(culturalName,loadedItemsList);
					}
				}
			}
		}
	}else {
		SplittedView* pGView = dynamic_cast<SplittedView*>(m_InnerView->geotimeView());
		if(pGView != nullptr){
			foreach (AbstractInnerView *p , pGView->getInnerViews()) {
				if (p == m_InnerView)
				{
					continue;
				}
				if ((p->viewType() == BasemapView ) || (p->viewType() == StackBasemapView ))
				{
					QGraphicsScene *scene = dynamic_cast<Abstract2DInnerView *>(p)->scene();
					if (scene)
					{
						dynamic_cast<GraphicSceneEditor *>(scene)->addLayer(culturalName,loadedItemsList);
					}
				}
			}
		}
	}
}



void GraphicSceneEditor::displayItemInfo()
{
    QGraphicsItem *selectedItem = selectedItems().first();
    QString Info;
    Info += ("Rotation : " + QString::number(selectedItem->rotation()) + "\n");
    Info += "Scene Coordinates Points : \n";
    foreach( QPointF p, dynamic_cast<GraphEditor_ItemInfo* >(selectedItem)->SceneCordinatesPoints()) {
        Info += ("{" + QString::number(p.x()) + "," + QString::number(p.y()) + "} ");
    }
    Info += "\nImage Coordinates Points : \n";
    foreach( QPointF p, dynamic_cast<GraphEditor_ItemInfo* >(selectedItem)->ImageCordinatesPoints()) {
        Info += ("{" + QString::number(p.x()) + "," + QString::number(p.y()) + "} ");
    }
    QMessageBox messageBox;
    messageBox.information(m_InnerView,"Item Info", Info);
}


QString GraphicSceneEditor::getUniqueName()
{
    QString filepath = m_InnerView->GraphicsLayersDirPath();
    QString name ="path3d_1";
    QString directory="3DPath/";
    QDir dir(filepath);
    bool res = dir.mkpath("3DPath");

    QFile file(filepath+directory+name+".2dp");
    int i =1;
    while(file.exists())
    {
        QString name ="path3d_"+QString::number(i);
        file.setFileName(filepath+directory+name+".2dp");

        i++;
    }

    return file.fileName();
}


QGraphicsItem* GraphicSceneEditor::findItem(QString id)
{
	 foreach(QGraphicsItem* p, selectedItems())
	{
		 GraphEditor_Item* item = dynamic_cast<GraphEditor_Item* >(p);
		if(item != nullptr && item->getID()== id)
		{
			return p;
		}
	}

	 qDebug()<<"findItem je retourne null ";
	 return nullptr;
}

GraphEditor_Path* GraphicSceneEditor::getCurrentBezier(QString nom )
{
	 for(std::map <int,QList<QGraphicsItem*>>::iterator it=m_GraphicItems.begin(); it!=m_GraphicItems.end(); ++it)
	 {
	        foreach (QGraphicsItem *p, it->second)
	        {

	        	GraphEditor_Path* path =  dynamic_cast<GraphEditor_Path* >(p);

	        	if(path != nullptr){
					if(path->getNameNurbs() == nom)
					{
						m_bezierSelected = path;
						return path;

					}
	        	}
	        }
	  }

	/*foreach(QGraphicsItem* p, selectedItems())
	{
		GraphEditor_Path* path =  dynamic_cast<GraphEditor_Path* >(p);
		if(path != nullptr && path->getNameNurbs() == nom)
		{
			m_bezierSelected = path;
			return path;
		}
	}*/
	return nullptr;
}

GraphEditor_Path* GraphicSceneEditor::getSelectBezier(QString nom )
{


	foreach(QGraphicsItem* p, selectedItems())
	{
		GraphEditor_Path* path =  dynamic_cast<GraphEditor_Path* >(p);
		if(path != nullptr && path->getNameNurbs() == nom)
		{
			m_bezierSelected = path;
			return path;
		}
	}
	return nullptr;
}

GraphEditor_RegularBezierPath* GraphicSceneEditor::findBezierPath()
{
	foreach(QGraphicsItem* p, selectedItems())
	{
		GraphEditor_RegularBezierPath* bezier =  dynamic_cast<GraphEditor_RegularBezierPath* >(p);
		if(bezier != nullptr) return bezier;
	}
	 return nullptr;
}

GraphEditor_Path* GraphicSceneEditor::findPath()
{
	foreach(QGraphicsItem* p, selectedItems())
	{
		GraphEditor_Path* path =  dynamic_cast<GraphEditor_Path* >(p);
		if(path != nullptr) return path;
	}
	return nullptr;
}




/*
void GraphicSceneEditor::createNurbs()
{
	NurbsWidget* nurbsWid = new NurbsWidget(nullptr);
	nurbsWid->show();
}*/

void GraphicSceneEditor::supprimerNurbs3d(QString name)
{

	if(m_InnerView != nullptr)
	{

		if(m_bezierSelected != nullptr)
		{
			deleteMyItem(m_SliceValue,m_bezierSelected);;


			//m_bezierSelected->deleteLater();
			//deleteItem(m_bezierSelected);

		}
		m_InnerView->deleteGeneratrice(name);;
	}
}

void GraphicSceneEditor::nurbs3d(QString name, QColor col)
{

	if(m_InnerView != nullptr)
	{
		QVector<QPointF>  listepoints;
		bool isOpen = true;

		bool withTangent = false;

		m_bezierSelected = findPath();

		if( m_bezierSelected != nullptr)
		{

			connect(m_bezierSelected,SIGNAL(BezierSelected(GraphEditor_Path*)),this, SLOT(onSelected(GraphEditor_Path*)));
			connect(m_bezierSelected,SIGNAL(BezierDeleted(QString)),this, SLOT(onDeleted(QString)));

			GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(m_bezierSelected);
			if(listBezier != nullptr)
			{

				connect(listBezier,SIGNAL(polygonChanged2(GraphEditor_ListBezierPath*)),this, SLOT(onPolygonChangedTangent(GraphEditor_ListBezierPath*)));

				//connect(listBezier,SIGNAL(polygonChanged(QVector<PointCtrl>, QVector<QPointF>,bool)),this, SLOT(onPolygonChangedTangent(QVector<PointCtrl>,QVector<QPointF>,bool)));
				withTangent = true;
				listepoints  = listBezier->getKeyPoints();


				/*
				QPointF normal = listBezier->getNormal(0);
				QVector3D nor(normal.x(),0.0f,normal.y());
				nor = nor.normalized();
				*/
				isOpen = !listBezier->isClosedPath();
			}
			else
			{
				connect(m_bezierSelected,SIGNAL(polygonChanged(QVector<QPointF>,bool)),this, SLOT(onPolygonChanged(QVector<QPointF>,bool)));
				listepoints  = m_bezierSelected->getKeyPoints();
				isOpen = !m_bezierSelected->isClosedPath();//isClosedCurve();
			}

		}
		else
		{
			QGraphicsItem *selectedItem = selectedItems().first();
			GraphEditor_EllipseShape* ellipse = dynamic_cast<GraphEditor_EllipseShape* >(selectedItem);
			if(ellipse != nullptr)
			{
				isOpen = false;
				listepoints =ellipse->getKeyPoints();
			}
			else
			{
				qDebug()<<" polyline est null donc pas de connect avec onPolygonChanged";
				listepoints =dynamic_cast<GraphEditor_ItemInfo* >(selectedItem)->SceneCordinatesPoints();
			}
			return;
		}
		QPointF cross;
		if(m_InnerView->viewType() == RandomView)
		{
			RandomLineView* random = dynamic_cast<RandomLineView* >(m_InnerView);
			if(random!= nullptr)
			{
				cross = random->m_position2DCross;
				random->setCurveBezier(m_SliceValue, m_bezierSelected);

			}
		}
		if(withTangent)
		{

			GraphEditor_ListBezierPath * listBezier = dynamic_cast<GraphEditor_ListBezierPath* >(m_bezierSelected);


			m_InnerView->setNurbsPoints(listBezier->GetListeCtrls(),listBezier, listepoints,name,isOpen,withTangent,col);
		}
		else
		{
			m_InnerView->setNurbsPoints(listepoints,isOpen,withTangent);
		}
	}
}


void GraphicSceneEditor::onPolygonChanged(QVector<QPointF> v, bool isopen)
{
	//m_InnerView->refreshNurbsPoints(v, isopen);
}

void GraphicSceneEditor::onPolygonChangedTangent(GraphEditor_ListBezierPath* path)
{
	QColor col;
	m_InnerView->refreshNurbsPoints(path,col);
}

void GraphicSceneEditor::onPolygonChangedTangent(QVector<PointCtrl> listeCtrl,QVector<QPointF> v, bool isopen)
{
	QColor col;

	QPointF cross;

	RandomLineView* random = dynamic_cast<RandomLineView* >(m_InnerView);
	if(random != nullptr )
	{
		cross = random->m_position2DCross;
		random->ResetLineOrtho();

	}

	m_InnerView->refreshNurbsPoints(listeCtrl,v, isopen,true,cross,col);
}

void GraphicSceneEditor::onSelected(GraphEditor_Path* p)
{

	m_bezierSelected = p;
	m_InnerView->setNurbsSelected(p->getNameId());
	// pOrthRandomView->m_BezierDirectrice =p;
}

void GraphicSceneEditor::directriceDeleted(QString name)
{
	if(m_bezierSelected)
		removeItem(m_bezierSelected);


}


void GraphicSceneEditor::onDeleted(QString name)
{

	m_bezierSelected = nullptr;
	// pOrthRandomView->m_BezierDirectrice =nullptr;
	m_InnerView->setNurbsDeleted(name);
}

void GraphicSceneEditor::refreshWidthOrtho(int widthO ,int width)
{

	m_randomWidth = width;
	emit updateWidthRandom(widthO, width);
}

void GraphicSceneEditor::path3d()
{

    //qDebug()<<" send path 3d"<<m_InnerView->GraphicsLayersDirPath();

    /*QString filepath = m_InnerView->GraphicsLayersDirPath();
    QString name ="path3d_1";



    QString directory="3DPath/";
    QDir dir(filepath);
    bool res = dir.mkpath("3DPath");*/



    QString outputFileName = getUniqueName();

    // create dir
    fs::path outputPath(outputFileName.toStdString());
    fs::path searchPath = outputPath.parent_path();
    bool dirExists = fs::exists(searchPath);
    bool valid = true;
    QStringList dirsToCreate;
    while (!dirExists && valid) {
        dirsToCreate.insert(0, QString(searchPath.filename().c_str()));
        valid = searchPath.has_parent_path();
        if (valid) {
            searchPath = searchPath.parent_path();
            dirExists = fs::exists(searchPath);
        }
    }
    if (dirExists && valid && dirsToCreate.count()>0) {
        QDir searchDir(QString(searchPath.c_str()));
        valid = searchDir.mkpath(dirsToCreate.join(QDir::separator()));
    }

    QFile file(outputFileName);
    if(!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug()<<" ouverture du fichier impossible :"<<file.fileName();
        return;
    }
    //qDebug()<<"chemin :"<<m_pathFiles<<directory<<"in2.txt";
    //qDebug()<<"nb element :"<<m_pathCamera.length();
    QTextStream out(&file);
    QGraphicsItem *selectedItem = selectedItems().first();
    GraphEditor_RegularBezierPath * bezierpath = dynamic_cast<GraphEditor_RegularBezierPath* >(selectedItem);
    QVector<QPointF>  listepoints;
    if( bezierpath != nullptr)
    {
        listepoints  = bezierpath->getKeyPoints();

    }
    else
    {
        listepoints =dynamic_cast<GraphEditor_ItemInfo* >(selectedItem)->SceneCordinatesPoints();
    //    qDebug()<<" path3d listepoints count:"<<listepoints.count();

    }


    foreach( QPointF p, listepoints)
    {
        out<<p.x()<<"|"<<p.y()<<"\n";
    }


    file.close();

}

void GraphicSceneEditor::cloneAndKeep()
{
	QGraphicsItem *selectedItem = selectedItems().first();

	if (selectedItem)
	{
		for (AbstractGraphicRep *pRep : m_InnerView->visibleReps() ) {
			if(dynamic_cast<iCUDAImageClone*>(pRep) != nullptr)
			{
				dynamic_cast<iCUDAImageClone*>(pRep)->cloneCUDAImageWithMask(selectedItem);
			}
		}
	}
}


BaseMapSurface* GraphicSceneEditor::cloneAndKeepFromLayerSpectrum() {
	QList<QGraphicsItem*> currentSelectedItems = selectedItems();
	if (currentSelectedItems.size()==0) {
		return nullptr;
	}
	QGraphicsItem *selectedItem = currentSelectedItems.first();
	BaseMapSurface *baseMapSurface = nullptr;

	if (selectedItem) {
		for (AbstractGraphicRep *pRep : m_InnerView->visibleReps() ) {
			// dynamic_cast may need to use type RGBLayerRGTRep instead of iCUDAImageCloneBaseMap
			if (dynamic_cast<iCUDAImageCloneBaseMap*>(pRep) != nullptr)
			{
				BaseMapSurface *new_Clone = dynamic_cast<iCUDAImageCloneBaseMap*>(pRep)->cloneCUDAImageWithMaskOnBaseMap(selectedItem);
				if (baseMapSurface)
				{
						bool ok = baseMapSurface->updateImages(new_Clone);
						if (!ok) {
							qDebug() << "GraphicSceneEditor::cloneAndKeepFromLayerSpectrum : Failed to fused two surfaces, keep the first one and dump the second one.";
						}
						new_Clone->deleteLater();
				}
				else
				{
					baseMapSurface = new_Clone;
				}
			}
		}
	}
	return baseMapSurface;
}

void GraphicSceneEditor::showItemsLayer(int  slicevalue)
{
    for(std::map <int,QList<QGraphicsItem*>>::iterator it=m_GraphicItems.begin(); it!=m_GraphicItems.end(); ++it)
    {
        if(it->first  == slicevalue)
        {
            foreach (QGraphicsItem *p, it->second) {
                p->show();
            }
        }
    }
}

QList<QGraphicsItem*> GraphicSceneEditor::CloneSceneItem(){
    return cloneItems(m_GraphicItems[m_SliceValue]);
}

void GraphicSceneEditor::hideItemsLayer(int  slicevalue)
{
    for(std::map <int,QList<QGraphicsItem*>>::iterator it=m_GraphicItems.begin(); it!=m_GraphicItems.end(); ++it)
    {
        if(it->first  == slicevalue)
        {
            foreach (QGraphicsItem *p, it->second) {
                if (p)
                    p->hide();
            }
        }
    }
}

void GraphicSceneEditor::updateSlice(int  new_value)
{
    m_SliceValue = new_value;
    for(std::map <int,QList<QGraphicsItem*>>::iterator it=m_GraphicItems.begin(); it!=m_GraphicItems.end(); ++it)
    {
        if(it->first  == new_value)
        {
            foreach (QGraphicsItem *p, it->second) {
                if (p)
                    p->show();
            }
        }
        else
        {
            foreach (QGraphicsItem *p, it->second) {
                if (p)
                    p->hide();
            }
        }
    }
}

void GraphicSceneEditor::setSliceValue(int  new_value)
{
    m_SliceValue = new_value;
}

