#include "randomlineview.h"
//#include "randomview.moc"
//#include "randomview.h"

#include <sstream>
#include <iostream>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QSlider>
#include <QSpinBox>
#include <QRegularExpressionValidator>
#include <QMenu>
#include <QPoint>
#include <QPointF>
#include <QRect>
#include <cmath>

#include "randomrep.h"
#include "sliceqglgraphicsview.h"
//#include "slicepositioncontroler.h"
#include "qglgridtickitem.h"
#include "cudaimagepaletteholder.h"
#include "mousetrackingevent.h"
#include "seismicsurvey.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "qglfixedaxisitem.h"
#include "affine2dtransformation.h"
#include "randomTransformation.h"
#include "statusbar.h"
#include "qglscalebaritem.h"
#include "qgllineitem.h"
//#include "isliceablerep.h"
//#include "LayerSpectrumDialog.h"
#include "affine2dtransformation.h"
#include "geometry2dtoolbox.h"
#include "baseqglgraphicsview.h"
#include "isampledependantrep.h"
#include "qglfixedaxisitem.h"
#include "multiseedrandomrep.h"
#include "GraphEditor_LineShape.h"
#include "mtlengthunit.h"


#include <QSlider>
#include <QSpinBox>

class RandomLineViewOverlayPainter : public AbstractOverlayPainter {
private:
    RandomLineView* m_view2d;
public:
    RandomLineViewOverlayPainter(RandomLineView* view2d) : m_view2d(view2d){
    }
    virtual ~RandomLineViewOverlayPainter(){
    }

    QRect computeBoundingBox(const QFontMetrics& metrics, const QStringList& list, int width){
        int rwidth = 0;
        int rheight = 0;
        for(const QString& info : list){
            QString elided = metrics.elidedText(info,Qt::TextElideMode::ElideRight,width);
            QRect bounding = metrics.boundingRect(elided);
            rwidth = std::max(rwidth,bounding.width());
            rheight += bounding.height();
        }
        return QRect(0,0,rwidth,rheight);
    }

    void paintOverlay(QPainter* painter, const QRectF& rect) override{

        if ( !m_view2d->m_isShowValues )
            return;

        int halfWidth = rect.width()/2;
        painter->setPen(Qt::yellow);

        int topLeft = 5;
        int topRight = 5;
        int bottomRight = rect.height()-5;
        int valueFillX = 0;
        int valueFillY = 0;
        int valueHeightMax = 0;
        const int valueBorder = 1;

        QFontMetrics metrics = painter->fontMetrics();

        // ================== VALUES ====================================================
        //        QGraphicsView* graphicsView =  this->view2d->getScene()->views().first();
        //        QWidget* viewport = this->view2d->view->viewport();
        //        QPointF mousePosition1 = graphicsView->mapFromScene(view2d->lastMousePositionWorld.toPoint());
        //        QString valueStr = QString::number(view2d->lastMousePositionImage.x()) + "/" + QString::number(view2d->lastMousePositionImage.y());
        //
        QPointF lastMousePosition = m_view2d->getLastMousePosition();
        QPoint mapPos = m_view2d->m_view->mapFromScene(lastMousePosition);
        QPoint globalPos = m_view2d->m_view->mapToGlobal(mapPos);
        if ( lastMousePosition.x() != -1 ) {
            std::vector<QString> vect;
            for (AbstractGraphicRep *rep : m_view2d->m_visibleReps) {
                RandomRep *slice = nullptr;
                if ( ! (slice = dynamic_cast<RandomRep*>(rep))) {
                    continue;
                }
                IData* ds = slice->data();
                QString name = ds->name();
                Seismic3DAbstractDataset* dataset = nullptr;
                if ( ! (dataset = dynamic_cast<Seismic3DAbstractDataset*>(ds))) {
                    continue;
                }
                qDebug() << "Data Name: " << name <<
                        "Data Type: " << dataset->type();

                if (IMouseImageDataProvider *provider =
                        dynamic_cast<IMouseImageDataProvider*>(rep)) {
                    IMouseImageDataProvider::MouseInfo info;
                    if (provider->mouseData(lastMousePosition.x(), lastMousePosition.y(), info)) {
                        std::stringstream ss;
                        //ss << std::fixed << std::setprecision(2);
                        if (info.values.size() < 1) {
                            continue;
                        }

                        QString valueStr1 = metrics.elidedText(QString::number(info.values[0]),
                                Qt::TextElideMode::ElideRight,halfWidth);
                        vect.push_back(valueStr1);
                        QRect boundingVal = metrics.boundingRect(valueStr1);
                        if ( valueFillX < boundingVal.width() + 2 * valueBorder) valueFillX = boundingVal.width() + 2 * valueBorder;
                        if ( valueHeightMax < boundingVal.width()) valueHeightMax = boundingVal.width();
                        valueFillY += boundingVal.height() +2 * valueBorder;
                    }
                }
            }
            // Line Pointer not actif, draw values on cursor move
            // ================== VALUES ====================================================
            int valuePosX= mapPos.x() + 5;
            int valuePosY = mapPos.y() + 5;
            painter->fillRect(valuePosX, valuePosY ,valueFillX, valueFillY, QColor(0,0,0,255));
            reverse(vect.begin(), vect.end());
            for(QString s:vect) {
                QRect boundingVal = metrics.boundingRect(s);
                painter->drawRect( valuePosX, valuePosY, valueFillX, boundingVal.height()+2*valueBorder);
                painter->drawText( valuePosX+1, valuePosY+boundingVal.height()+valueBorder,s);
                valuePosY += boundingVal.height() + 2 * valueBorder;
            }
        }
        //
        //        // INFO ===============================================================================================
        //        for(auto pair : view2d->handleTovisuals){
        //            // ======= top left
        //            QStringList topLeftInfo = pair.second->topLeftInfo();
        //            QRect topLeftBounding = computeBoundingBox(metrics, topLeftInfo, halfWidth);
        //            painter->fillRect(0,topLeft,topLeftBounding.width()+10, topLeftBounding.height()+5,QColor(127,127,127,127));
        //            for(const QString& info : topLeftInfo){
        //                QString elided = metrics.elidedText(info,Qt::TextElideMode::ElideRight,halfWidth);
        //                QRect bounding = metrics.boundingRect(elided);
        //                topLeft += bounding.height();
        //                painter->drawText(5,topLeft,elided);
        //            }
        //
        //            // =========== top right
        //            QStringList topRightInfo = pair.second->topRightInfo();
        //            QRect topRightBounding = computeBoundingBox(metrics, topRightInfo, halfWidth);
        //            painter->fillRect(rect.width()-topRightBounding.width()-10,topRight,topRightBounding.width()+10, topRightBounding.height()+5,QColor(127,127,127,127));
        //            for(const QString& info : topRightInfo){
        //                QString elided = metrics.elidedText(info,Qt::TextElideMode::ElideRight,halfWidth);
        //                QRect bounding = metrics.boundingRect(elided);
        //                int xoff = rect.width()-bounding.width()-5;
        //                topRight += bounding.height();
        //                painter->drawText(xoff, topRight,elided);
        //            }
        //
        //            // ====== bottom right
        //            QStringList bottomRightInfo = pair.second->bottomRightInfo();
        //            QRect bottomRightBounding = computeBoundingBox(metrics, bottomRightInfo, rect.width());
        //            painter->fillRect(rect.width()-bottomRightBounding.width()-10,bottomRight,bottomRightBounding.width()+10, bottomRightBounding.height()+5,QColor(127,127,127,127));
        //            for(const QString& info : bottomRightInfo){
        //                QString elided = metrics.elidedText(info,Qt::TextElideMode::ElideLeft,rect.width());
        //                QRect bounding = metrics.boundingRect(elided);
        //                int xoff = rect.width()-bounding.width()-5;
        //                bottomRight -= bounding.height();
        //                painter->fillRect(xoff,bottomRight,bounding.width(), bounding.height(),QColor(127,127,127,127));
        //            }
        //        }
    }
};

RandomLineView::RandomLineView(QPolygonF polyLine,ViewType type, QString uniqueName,eRandomType eType):
                        Abstract2DInnerView(false, new SliceQGLGraphicsView(),new SliceQGLGraphicsView(),new SliceQGLGraphicsView(), uniqueName)
{
    m_viewType = type;
    m_polyLine = polyLine;
    m_randomType = SampleUnit::NONE;
    m_eType = eType;
    m_slice = nullptr;
    m_UpdateRep = false;
    m_ItemSection = nullptr;
    //    if (type == ViewType::InlineView) {
    //        statusBar()->setWorldCoordinateLabels("Xline", "Depth");
    //    } else if (type == ViewType::XLineView) {
    //        statusBar()->setWorldCoordinateLabels("Inline", "Depth");
    //    }

    m_verticalAxis = nullptr;
    m_horizontalAxis = nullptr;
    m_RandomOrthogonal = nullptr;

    // Overlay for values on cursor position

    QWidget *sliceBox = createSliceBox(windowTitle());
    sliceBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_mainLayout->insertWidget(0,sliceBox);

    m_overlayPainter = new RandomLineViewOverlayPainter(this);
    m_view->addOverlayPainter(m_overlayPainter);



    connect(m_scene, &QGraphicsScene::sceneRectChanged, this, &RandomLineView::mainSceneRectChanged);
}

eRandomType RandomLineView::getRandomType()
{
    return m_eType;
}

void RandomLineView::setPolyLine(QPolygonF polyLine){
    m_polyLine = polyLine;
    m_UpdateRep = true;

    RandomRep *slice = firstRandom();
     if (slice != nullptr)
     {
    	 Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) slice->data();
    	 discreatePoly(dataset);


     }
     else
     {
    	 m_discreatePolyLine.clear();
     }



    for(int i=0;i< m_visibleReps.size();i++)
    {
    	 AbstractGraphicRep *r = m_visibleReps[i];
    	RandomRep* pRandomRep = dynamic_cast<RandomRep*>(r);
    	if(pRandomRep != nullptr){
    		pRandomRep->setPolyLine(polyLine);
    	}

    }
/*

    // RandomRep first
    for(int i = m_visibleReps.size()-1 ;i>=0;i--){
        AbstractGraphicRep *r = m_visibleReps[i];
        RandomRep* pRandomRep = dynamic_cast<RandomRep*>(r);
        if(pRandomRep != nullptr){
            hideRep(pRandomRep);
            pRandomRep->deleteLayer();
            pRandomRep->setUpdatedFlag(true);
            showRep(pRandomRep);
            pRandomRep->setUpdatedFlag(false);
        }
    }
*/
    // Other rep later
    for(int i = m_visibleReps.size()-1 ;i>=0;
    		i--){
        AbstractGraphicRep *r = m_visibleReps[i];
        RandomRep* pRandomRep = dynamic_cast<RandomRep*>(r);
        if(pRandomRep == nullptr){
            hideRep(r);
            r->deleteLayer();
            showRep(r);
        }
    }


    emit updateOrtho(m_polyLine);


}

void RandomLineView::setOrthogonalView(RandomLineView *pOrthoView){

    if(pOrthoView != nullptr){
        m_RandomOrthogonal = pOrthoView;
        // La directrice
        if(m_eType != eTypeOrthogonal){
            if(m_slice->image() != nullptr){
            	int departX = m_slice->image()->worldExtent().x();
            	int width = m_slice->image()->worldExtent().width();
                m_RandomOrthogonal->defineSliceMinMax(departX, width) ;//m_slice->image());
                m_RandomOrthogonal->setOrthoWorldDiscreatePolyline(m_worldDiscreatePolyLine);
                m_RandomOrthogonal->setOrthoDiscreatePolyline(m_discreatePolyLine);



                initTransformation();

            }
        }
        else
        {
        	/*if(m_lineVItem==nullptr)
        	{
        		int posX = m_slice->image()->worldExtent().left()+ m_slice->image()->worldExtent().width()*0.5f;
        		int posY = m_slice->image()->worldExtent().top()+m_slice->image()->worldExtent().height()*0.5f;
        		int width = m_slice->image()->worldExtent().width()*0.5f;
        		int height = m_slice->image()->worldExtent().height()*0.5f;

        		qDebug()<<" posX : "<<posX<<" , posY :"<<posY<<" , Width : "<<width<<" , height :"<<height;
        	//	int posX =0;
        	//	int posY =0;
        		m_lineVItem = new QGraphicsLineItem(posX-200,posY-200,posX+200,posY+200);
        		m_lineVItem->setZValue(CROSS_ITEM_Z);
        		QPen pen(Qt::blue);
        		pen.setWidth(3);
        		pen.setCosmetic(true);
        		m_lineVItem->setPen(pen);

        		m_lineHItem = new QGraphicsLineItem(posX-200,posY+200,posX+200,posY-200);
        		m_lineHItem->setZValue(CROSS_ITEM_Z);
        		m_lineHItem->setPen(pen);

        		m_scene->addItem(m_lineVItem);
        		m_scene->addItem(m_lineHItem);

        		//	m_lineHItem = new QGLLineItem(m_horizontalAxis->boundingRect(),image,QGLLineItem::Direction::HORIZONTAL);



        		mHeight = m_verticalAxis->boundingRect().height();

        	}*/
       }
    }
}

RandomLineView * RandomLineView::getOrthogonalView(void){
    return m_RandomOrthogonal;
}

QWidget* RandomLineView::createSliceBox(const QString &title) {
    QWidget *controler = new QWidget(this);

    QGridLayout *gridLayout = new QGridLayout(controler);
    gridLayout->setContentsMargins(0, 0, 0, 0);
    gridLayout->addWidget(new QLabel(title), 1, 0, 1, 1);

    m_sliceImageSlider = new QSlider();
    m_sliceImageSlider->setOrientation(Qt::Horizontal);
    m_sliceImageSlider->setSingleStep(1);
    m_sliceImageSlider->setTracking(false);//sylvain true
    m_sliceImageSlider->setTickInterval(10);
    m_sliceImageSlider->setMinimum(0);
    m_sliceImageSlider->setMaximum(1);
    m_sliceImageSlider->setValue(0);

    gridLayout->addWidget(m_sliceImageSlider, 1, 5, 1, 1);

    m_sliceSpin = new QSpinBox();
    m_sliceSpin->setMinimum(0);
    m_sliceSpin->setMaximum(1);
    m_sliceSpin->setSingleStep(1);
    m_sliceSpin->setValue(0);
    m_sliceSpin->setWrapping(false);

    gridLayout->addWidget(m_sliceSpin, 1, 2, 1, 1);



    if(m_eType == eTypeOrthogonal ){
        m_Orthogonalwidth = new QLineEdit("3000");
        m_Orthogonalwidth->setMaximumWidth(100);
        m_Orthogonalwidth->setMaxLength(10);
        m_Orthogonalwidth->setValidator(new QRegularExpressionValidator(QRegularExpression("[0-9]*"), m_Orthogonalwidth));
        gridLayout->addWidget(m_Orthogonalwidth, 1, 1, 1, 1);
        connect(m_Orthogonalwidth,SIGNAL(textChanged(QString)),this,SLOT(updateOrthogonalWidth(QString)));

        m_playButton = new QToolButton();
        m_playButton->setIcon(style()->standardPixmap( QStyle::SP_MediaPlay));
        m_playButton->setToolTip("move X section");
        m_playButton->setCheckable(true);

        m_playButtonInverse = new QToolButton();


        m_playButtonInverse->setIcon(QIcon(QString(":/slicer/icons/playinverse.png")));
        m_playButtonInverse->setToolTip("move X section");
        m_playButtonInverse->setCheckable(true);

        QToolButton *m_previousButton = new QToolButton();
        m_previousButton->setIcon(style()->standardPixmap( QStyle::SP_MediaSkipBackward));// QStyle::SP_MediaSkipBackward
        m_previousButton->setToolTip("move previous key points");

        QToolButton *m_nextButton = new QToolButton();
		m_nextButton->setIcon(style()->standardPixmap( QStyle::SP_MediaSkipForward));
		m_nextButton->setToolTip("move next key points");

		gridLayout->addWidget(m_previousButton, 1,3, 1, 1);
		gridLayout->addWidget(m_playButtonInverse, 1,4, 1, 1);
        gridLayout->addWidget(m_playButton, 1, 6, 1, 1);
        gridLayout->addWidget(m_nextButton, 1,7, 1, 1);

        connect(m_previousButton,SIGNAL(clicked()), this,SLOT(previousSection()));
        connect(m_playButton,SIGNAL(clicked()), this,SLOT(playMoveSection()));
        connect(m_playButtonInverse,SIGNAL(clicked()), this,SLOT(playInverseMoveSection()));
        connect(m_nextButton,SIGNAL(clicked()), this,SLOT(nextSection()));

        mTimer = new QTimer();
        connect(mTimer,SIGNAL(timeout()), this,SLOT(onTimeout()));

    }

    connect(m_sliceImageSlider, SIGNAL(valueChanged(int)), this,SLOT(sliceFinishChanged(int)));
    connect(m_sliceImageSlider, SIGNAL(sliderMoved(int)), this,SLOT(sliceChanged(int)));
    connect(m_sliceImageSlider, SIGNAL(sliderReleased()), this,SLOT(sliceFinish()));

    connect(m_sliceSpin, SIGNAL(valueChanged(int)), this,SLOT(sliceChanged(int )));

    return controler;
}

void RandomLineView::updateOrthogonalWidth(const QString &newText){
    double newValue = newText.toDouble();
    if(newValue > 100){ // TODO : workaround:minor value crash program to fix later
        emit newWidthOrthogonal(newValue);


        //dynamic_cast<GraphicSceneEditor*>(m_scene)->refreshWidthOrtho(m_WidthOriginal,m_rectIntersect.width());
        emit newWidthOrtho(m_polyLine);

        m_coef = newValue/3000.0f;


        dynamic_cast<GraphicSceneEditor*>(m_scene)->refreshWidthOrtho(m_WidthOriginal,m_rectIntersect.width()*m_coef);

        m_WidthOriginal = m_rectIntersect.width()*m_coef;

/*
        qDebug()<<m_WidthOriginal<<" : m_WidthOriginal,m_coef:"<<m_coef;

        QPointF pos2D  = m_posTemp;//     absoluteWorldToViewWorldExtended(m_posTemp);

        pos2D.setX( pos2D.x()*m_coef);
        m_position2DCross = pos2D;

*/

        QVector3D posG(m_posTemp.x(),0.0f, m_posTemp.y());

		QPointF pos2D  = absoluteWorldToViewWorldExtended(posG);
		float altY = m_isoBuffer.getAltitude(m_posTemp);

		QPointF position(pos2D.x(), altY);


        m_crossItem->setPos(position);




    }
}

void RandomLineView::setCrossPosition(QPointF pos)
{
	//float altY = getHorizonBuffer()->getAltitude(pos);

	//QPointF position(pos.x(), altY);

	//if(m_isoBuffer == nullptr) qDebug()<<" nullptr sur mon buffer";

	QVector3D posG(pos.x(),0.0f, pos.y());

	QPointF pos2D  = absoluteWorldToViewWorldExtended(posG);
	float altY = m_isoBuffer.getAltitude(pos);

	QPointF position(pos2D.x(), altY);

	m_posTemp = pos;
	//qDebug()<<" m_posTemp"<<m_posTemp;

  	m_crossItem->setPos(position);
}

void RandomLineView::nurbsXYZChanged(QVector3D pos)
{


	if(m_crossItem!= nullptr)
	{
		m_position3DCross = pos;

		//m_posTemp = pos;
		QPointF pos2D  = absoluteWorldToViewWorldExtended(pos);

		m_position2DCross = pos2D;

	//	qDebug()<<"m_position2DCross:  "<<m_position2DCross;

		m_crossItem->setPos(pos2D);
		m_crossItem->show();

	}

/*	if(m_lineHItem!= nullptr)
		{
			m_lineHItem->show();
			m_lineVItem->show();


			m_posTemp = pos;
			QPointF pos2D  = absoluteWorldToViewWorldExtended(pos);


			m_lineHItem->setLine(pos2D.x()-m_sizeCross,pos2D.y()+m_sizeCross,pos2D.x()+m_sizeCross,pos2D.y()-m_sizeCross);
			m_lineVItem->setLine(pos2D.x()-m_sizeCross,pos2D.y()-m_sizeCross,pos2D.x()+m_sizeCross,pos2D.y()+m_sizeCross);//posX-30,posY-30,posX+30,posY+30);
		}*/
}

void RandomLineView::nurbYChanged(float value)
{
	/*if(m_lineHItem!= nullptr)
	{
		m_lineHItem->show();
		m_lineVItem->show();
		int posX =m_rectIntersect.left()+ m_rectIntersect.width()*m_coef*0.5f;
		int posY = m_rectIntersect.top()+m_rectIntersect.height()*(1.0f-value);

	//	m_lineHItem->setLine(posX-m_sizeCross,posY+m_sizeCross,posX+m_sizeCross,posY-m_sizeCross);
	//	m_lineVItem->setLine(posX-m_sizeCross,posY-m_sizeCross,posX+m_sizeCross,posY+m_sizeCross);//posX-30,posY-30,posX+30,posY+30);
	}*/

}
void RandomLineView::setColorCross(QColor col)
{
	if(m_crossItem!= nullptr)
	{
		m_crossItem->setColor(col);
	}
	if(m_bezierItem != nullptr)
	{
		m_bezierItem->setColor(col);
	}
	/*else
	{
		qDebug()<<" m_bezierItem est nullllll";
	}*/

	//dynamic_cast<GraphicSceneEditor*>(m_scene)->getCurrentBezier(name);

/*	QPen pen(m_lineVItem->pen());
	pen.setColor(col);
	m_lineVItem->setPen(pen);
	m_lineHItem->setPen(pen);*/
}

void RandomLineView::curveChanged(std::vector<QVector3D> listepts3d, bool isopen)
{


	dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteItem();
	QPolygonF poly;
	for(int i=0;i<listepts3d.size();i++)
	{
		//qDebug()<<" listepts3d[i] :"<<listepts3d[i];
		QPointF pt = absoluteWorldToViewWorldExtended(listepts3d[i]);
		poly.push_back(pt);
		//qDebug()<<" pt :"<<pt;

	}


	if(isopen==false) poly.push_back(poly[0]);

	if(m_curveBezier != nullptr)
	{
		//m_curveBezier->hide();
		dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteMyItem(m_numSlice,m_curveBezier);

		m_curveBezier = nullptr;
	}


	if(m_curveItem != nullptr)dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteItem(m_curveItem);
	m_curveItem = dynamic_cast<GraphicSceneEditor*>(m_scene)->addNewCurve(poly,isopen);
	connect(m_curveItem,SIGNAL(currentIndexChanged(int)),this,SLOT(receiveCurrentIndex(int)));

}


void RandomLineView::curveChangedTangent2(QVector<QVector3D> listepts3d, QVector<QVector3D> listetan3d,bool isopen,QPointF cross,QString nameNurbs)
{
/*	qDebug()<<"liste generatrice avant "<<listepts3d.count();
		for(int i=0;i<listepts3d.count();i++)
		{
			qDebug()<<"pos :"<<listepts3d[i];
		}*/

	QVector<PointCtrl> listepts;

	int indexTan = 0;

	for(int i=0;i<listepts3d.count();i++)
	{
		QPointF pos= absoluteWorldToViewWorldExtended(listepts3d[i]);
		QPointF tan1= absoluteWorldToViewWorldExtended( listetan3d[indexTan]);
		indexTan++;
		QPointF tan2= absoluteWorldToViewWorldExtended( listetan3d[indexTan]);
		indexTan++;
		QPointF ctrl1( tan1);
		QPointF ctrl2( tan2);
		listepts.push_back(PointCtrl(pos,ctrl1, ctrl2 ));
	}


/*	qDebug()<<"liste generatrice apres "<<listepts.count();
	for(int i=0;i<listepts.count();i++)
	{
		qDebug()<<"pos :"<<listepts[i].m_position;
	}*/

	dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteItem();


	if(m_curveBezier != nullptr)
	{
		//m_curveBezier->hide();
		dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteMyItem(m_numSlice,m_curveBezier);

		m_curveBezier = nullptr;
	}

	if(m_bezierItem != nullptr)dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteItem(m_bezierItem);
	m_bezierItem = dynamic_cast<GraphicSceneEditor*>(m_scene)->addNewCurve(listepts,isopen,nameNurbs);
	connect(m_bezierItem,SIGNAL(currentIndexChanged(int)),this,SLOT(receiveCurrentIndex(int)));



}



void RandomLineView::curveChangedTangent(QVector<PointCtrl> listepts3d, bool isopen,QPointF cross)
{
	dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteItem();

	if(m_curveBezier != nullptr)
	{
		//m_curveBezier->hide();
		dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteMyItem(m_numSlice,m_curveBezier);

		m_curveBezier = nullptr;
	}



	QPointF decal = m_position2DCross - cross;
	//qDebug()<< "cross :"<< cross <<" decal==> "<<decal;
	//QVector<PointCtrl> listepts3dTr;
	for(int i=0;i<listepts3d.count();i++)
	{
		listepts3d[i].m_position = listepts3d[i].m_position +decal;
		listepts3d[i].m_ctrl1 = listepts3d[i].m_ctrl1 +decal;
		listepts3d[i].m_ctrl2 = listepts3d[i].m_ctrl2 +decal;
	}


	if(m_bezierItem != nullptr)dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteItem(m_bezierItem);
	m_bezierItem = dynamic_cast<GraphicSceneEditor*>(m_scene)->addNewCurve(listepts3d,isopen,"");
	connect(m_bezierItem,SIGNAL(currentIndexChanged(int)),this,SLOT(receiveCurrentIndex(int)));

}

void RandomLineView::setItemSection(GraphEditor_ItemInfo* pItem)
  {
  	if(m_ItemSection != nullptr)
  	{
  		QObject* obj = dynamic_cast<QObject*>(m_ItemSection) ;
  		disconnect(obj,SIGNAL(destroyed()),this,SLOT(resetItemSection()));
  	}

  	m_ItemSection = pItem ;
  	QObject* obj = dynamic_cast<QObject*>(m_ItemSection) ;
  	if(obj!= nullptr)
  	{
  		connect(obj,SIGNAL(destroyed()),this,SLOT(resetItemSection()));
  	}


  }


void RandomLineView::curveChangedTangentOpt(GraphEditor_ListBezierPath* path)
{
	//qDebug()<<"debut  RandomLineView::curveChangedTangentOpt : "<<path->GetListeCtrls().count();
	QVector<PointCtrl> listepts3d = path->GetListeCtrls();
	bool isopen = !path->isClosedPath();
	dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteItem();

	if(m_curveBezier != nullptr)
	{
		//m_curveBezier->hide();
		dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteMyItem(m_numSlice,m_curveBezier);

		m_curveBezier = nullptr;
	}

/*
	QPointF decal = m_position2DCross - cross;
	//qDebug()<< "cross :"<< cross <<" decal==> "<<decal;
	//QVector<PointCtrl> listepts3dTr;
	for(int i=0;i<listepts3d.count();i++)
	{
		listepts3d[i].m_position = listepts3d[i].m_position +decal;
		listepts3d[i].m_ctrl1 = listepts3d[i].m_ctrl1 +decal;
		listepts3d[i].m_ctrl2 = listepts3d[i].m_ctrl2 +decal;
	}
*/

	//qDebug()<<" RandomLineView::curveChangedTangentOpt : "<<path->GetListeCtrls().count();

	if(m_bezierItem != nullptr)dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteItem(m_bezierItem);

	m_bezierItem = dynamic_cast<GraphicSceneEditor*>(m_scene)->addNewCurve(listepts3d,isopen,path->getNameNurbs());
	connect(m_bezierItem,SIGNAL(currentIndexChanged(int)),this,SLOT(receiveCurrentIndex(int)));

}

void RandomLineView::deleteCurves()
{
	dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteMyItem(m_numSlice,m_curveBezier);
	dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteMyItem(m_numSlice,m_curveItem);
	dynamic_cast<GraphicSceneEditor*>(m_scene)->deleteMyItem(m_numSlice,m_bezierItem);
	m_curveBezier = nullptr;
	m_curveItem = nullptr;
	m_bezierItem = nullptr;


	if(m_ItemSection != nullptr)
	{

		delete m_ItemSection;
		m_ItemSection = nullptr;
	}
	/*if(m_curveBezier != nullptr)
	{
		delete m_curveBezier;
		m_curveBezier = nullptr;
	}

	if(m_curveItem != nullptr)
	{
		delete m_curveItem;
		m_curveItem = nullptr;
	}*/
}

void RandomLineView::receiveCurrentIndex(int index)
{
	if( m_curveItem->pen().style() != Qt::SolidLine)
	{
		 QPen Pen(QColor(255,255,255,255), 3, Qt::SolidLine);
		 Pen.setCosmetic(true);
		 m_curveItem->setPen(Pen);

	}
	emit sendIndexChanged(index);
}


void RandomLineView::defineSliceMinMax(int departX, int width ) {//IGeorefImage *image) {

	QSignalBlocker b1(m_sliceImageSlider);

	    m_sliceImageSlider->setMinimum(departX);
	    m_sliceImageSlider->setMaximum(departX + width);
	    m_sliceImageSlider->setSingleStep(1);
	    int pageStep = (int) (departX + width);
	    m_sliceImageSlider->setPageStep(pageStep);
	    m_sliceImageSlider->setTickInterval(1);
	    //
	    QSignalBlocker b2(m_sliceSpin);
	    m_sliceSpin->setMinimum(departX);
	    m_sliceSpin->setMaximum(departX +width);
	    //    dynamic_cast<GraphicSceneEditor *> (m_scene)->setSliceValue((int) image->worldExtent().x());
	    m_sliceSpin->setSingleStep(1);


  /*  QSignalBlocker b1(m_sliceImageSlider);

    m_sliceImageSlider->setMinimum((int) (image->worldExtent().x()));
    m_sliceImageSlider->setMaximum((int) (image->worldExtent().x() + image->worldExtent().width()));
    m_sliceImageSlider->setSingleStep(1);
    int pageStep = (int) ((image->worldExtent().x() + image->worldExtent().width())/1);
    m_sliceImageSlider->setPageStep(pageStep);
    m_sliceImageSlider->setTickInterval(1);
    //
    QSignalBlocker b2(m_sliceSpin);
    m_sliceSpin->setMinimum((int) (image->worldExtent().x()));
    m_sliceSpin->setMaximum((int) (image->worldExtent().x() + image->worldExtent().width()));
    //    dynamic_cast<GraphicSceneEditor *> (m_scene)->setSliceValue((int) image->worldExtent().x());
    m_sliceSpin->setSingleStep(1);*/
}

void RandomLineView::defineSliceVal(int image) {
    QSignalBlocker b1(m_sliceImageSlider);
    m_sliceImageSlider->setValue(image);

    QSignalBlocker b2(m_sliceSpin);
    m_sliceSpin->setValue(image);
}

void RandomLineView::updateSlicePosition(int worldVal) {
    defineSliceVal(worldVal);
}

void RandomLineView::sliceChanged(int val) {
    updateSlicePosition(val);

    m_sliceValueWorld = val;
    if(m_eType == eTypeOrthogonal){
        emit orthogonalSliceMoved(val,m_sliceImageSlider->maximum(),m_currentName);
        getPositionMap(val);


     //   float m_coefPosition = (float)(val/(float)m_sliceImageSlider->maximum());
    //    qDebug()<<" emit orthogonalSliceMoved "<<val;
    //     emit dynamic_cast<GraphicSceneEditor *> (m_scene)->movePosition(m_coefPosition);

    }
}
void RandomLineView::sliceFinish()
{
	emit createXSection3D();
}

void RandomLineView::showEvent(QShowEvent* event)
{
//	qDebug()<<" getBaseTitle : "<<getBaseTitle();
	if(m_BezierDirectrice != nullptr)
	{

		QString nom = getBaseTitle();
		nom = nom.replace("Random-","");
	//	qDebug()<<" je selectionne  "<<nom;
		m_currentName = nom;

		//m_BezierDirectrice->selected(true);
	}
}


void RandomLineView::previousSection()
{

	if( m_BezierDirectrice == nullptr )
	{
		qDebug()<<"m_BezierDirectrice est NULLLLLLL ";
		return;
	}
	QVector<QPointF> keyPoints =  m_BezierDirectrice->getKeyPoints();

	float distanceTotal = 0.0f;
	float dist= 0.0f;
	float coef = -1.0f;
	for(int i =0;i<keyPoints.count()-1;i++)
	{
		double d = sqrt( pow( (keyPoints[i].x() - keyPoints[i+1].x()), 2) + pow( (keyPoints[i].y() - keyPoints[i+1].y()), 2));

		distanceTotal +=d;
	}

	for(int i =keyPoints.count()-1;i>0;i--)
	{
		double d = getDistance(i,keyPoints);
		//double d = sqrt( pow( (keyPoints[0].x() - keyPoints[i].x()), 2) + pow( (keyPoints[0].y() - keyPoints[i].y()), 2));

		 dist = d;
		float coefcurrent =(float)( m_sliceImageSlider->value() /(float)m_sliceImageSlider->maximum());
		float nextCoef = dist/distanceTotal;


		if( coef < 0.0f && nextCoef  <  coefcurrent- 0.005f )
		{
			coef = nextCoef;
		}
	}

	m_sliceImageSlider->setValue(coef * m_sliceImageSlider->maximum());
	emit createXSection3D();
}

void RandomLineView::nextSection()
{

	if( m_BezierDirectrice == nullptr )
	{
		qDebug()<<"m_BezierDirectrice est NULLLLLLL ";
		return;
	}
	QVector<QPointF> keyPoints =  m_BezierDirectrice->getKeyPoints();

	float distanceTotal = 0.0f;
	float dist= 0.0f;
	float coef = -1.0f;
	for(int i =0;i<keyPoints.count()-1;i++)
	{
		double d = sqrt( pow( (keyPoints[i].x() - keyPoints[i+1].x()), 2) + pow( (keyPoints[i].y() - keyPoints[i+1].y()), 2));

		distanceTotal +=d;
	}

	for(int i =1;i<keyPoints.count();i++)
	{
		double d = getDistance(i,keyPoints);
		//double d = sqrt( pow( (keyPoints[0].x() - keyPoints[i].x()), 2) + pow( (keyPoints[0].y() - keyPoints[i].y()), 2));

		 dist = d;
		float coefcurrent =(float)( m_sliceImageSlider->value() /(float)m_sliceImageSlider->maximum());
		float nextCoef = dist/distanceTotal;


		if( coef < 0.0f && nextCoef  >  coefcurrent+ 0.005f )
		{
			coef = nextCoef;
		}
	}

	m_sliceImageSlider->setValue(coef * m_sliceImageSlider->maximum());

	emit createXSection3D();


}

float  RandomLineView::getDistance(int index,QVector<QPointF> keyPoints)
{
	float distanceTotal = 0.0;


	for(int i =0;i<index;i++)
	{
		double d = sqrt( pow( (keyPoints[i].x() - keyPoints[i+1].x()), 2) + pow( (keyPoints[i].y() - keyPoints[i+1].y()), 2));

		distanceTotal +=d;
	}
	return distanceTotal;
}

void RandomLineView::playMoveSection()
{
	if(m_playButton->isChecked())
	{
		m_sensAnim= 1;
		m_animActif = true;
		mTimer->start(50);
	}
	else
	{
		m_animActif= false;
		mTimer->stop();
	}

	emit etatAnimationChanged(m_animActif);
}

void RandomLineView::playInverseMoveSection()
{
	if(m_playButtonInverse->isChecked())
	{
		m_sensAnim= -1;
		m_animActif = true;
		mTimer->start(50);
	}
	else
	{
		m_animActif= false;
		mTimer->stop();
	}

	emit etatAnimationChanged(m_animActif);
}



void RandomLineView::onTimeout()
{
	if(m_sensAnim> 0)
	{
		if(m_sliceImageSlider->value()-m_stepsAnimation <= m_sliceImageSlider->maximum())
			m_sliceImageSlider->setValue( m_sliceImageSlider->value() + m_stepsAnimation );
		else
		{
			mTimer->stop();
			m_animActif= false;

			emit etatAnimationChanged(m_animActif);
		}
	}
	else
	{
		if(m_sliceImageSlider->value()+m_stepsAnimation >= 0)
			m_sliceImageSlider->setValue( m_sliceImageSlider->value() - m_stepsAnimation );
		else
		{
			mTimer->stop();
			m_animActif= false;
			emit etatAnimationChanged(m_animActif);
		}
	}
}


void RandomLineView::refreshOrtho(QVector3D pos, QVector3D normal)
{
	//getPositionMap(m_sliceValueWorld);


	QPointF pos2D( pos.x(),pos.z());
	emit updateOrthoFrom3D(normal,pos2D);
}


int RandomLineView::getAltitudeCross()
{

}

void RandomLineView::setSpeedAnimation(int val)
{
	m_stepsAnimation = val;
}

void RandomLineView::sliceFinishChanged(int val) {
    updateSlicePosition(val);



    m_sliceValueWorld = val;
    if(m_eType == eTypeOrthogonal){
        emit orthogonalSliceMoved(val,m_sliceImageSlider->maximum(),m_currentName);
       // qDebug()<<" emit orthogonalSliceMoved ";
        getPositionMap(val);


    }
}

void RandomLineView::orthogonalMoved(int val) {
    if(m_eType == eTypeStandard){
        //std::cout << "ORTHOGONAL MOVED " << val << "\n";
        getPositionMap(val);
        emit newPointPosition(m_pointOthogonal);


    }
}

QVector<RandomRep*> RandomLineView::getRandomVisible()
{
	QVector<RandomRep*> lesReps;

	for (AbstractGraphicRep *r : m_visibleReps) {
		if (RandomRep *slice = dynamic_cast<RandomRep*>(r)) {
			lesReps.push_back(slice);
		}
	}
	return lesReps;
}

RandomRep* RandomLineView::firstRandom() const {
    int count = 0;
    for (AbstractGraphicRep *r : m_visibleReps) {
        if (RandomRep *slice = dynamic_cast<RandomRep*>(r)) {
            return slice;
        }
    }
    return nullptr;
}

RandomRep* RandomLineView::lastRandom() const {
    int count = 0;
    for (long i=m_visibleReps.count()-1; i>=0; i--) {
        AbstractGraphicRep *r = m_visibleReps[i];
        if (RandomRep *slice = dynamic_cast<RandomRep*>(r)) {
            return slice;
        }
    }
    return nullptr;
}

RandomRep* RandomLineView::getRandomSismic() const
{
	RandomRep* res = nullptr;
	  for (AbstractGraphicRep *r : m_visibleReps) {
	        if (RandomRep *slice = dynamic_cast<RandomRep*>(r))
	        {
	        	if( res == nullptr) res = slice;
	        	if(slice->getdataset()->type() == Seismic3DAbstractDataset::Seismic)
	        	{
	        		res =  slice;
	        		break;
	        	}
	        }
	    }
	    return res ;
}


long RandomLineView::getDiscreatePolyLineIndexFromScenePos(QPointF scenePos) {
    double val = std::round(scenePos.x()); // round to get the point
    long out;
    if (val>=0 && val<m_discreatePolyLine.size()) {
        out = val;
    } else {
        out = -1;
    }
    return out;
}

QPointF RandomLineView::getScenePosFromDiscreatePolyLinePos(QPointF imagePoint) {
    QPointF out;
    long x = std::round(imagePoint.x());
    if (x>=0 && x<m_discreatePolyLine.size()) {
        out = imagePoint; // for now there is no transformation to be done
    }
    return out;
}

std::tuple<long, double, bool> RandomLineView::getDiscreatePolyLineIndexFromWorldPos(QPointF worldPos) {
    RandomRep *rep = firstRandom();
    if (rep == nullptr)
        return std::tuple<long, double, bool>(0, 0, false);

    double imageI, imageJ;

    Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
    SeismicSurvey *survey = dataset->survey();
    //    int pos = m_currentSliceWorldPosition;
    std::size_t indexNearestPoint;
    double distance = std::numeric_limits<double>::max();

    dataset->ijToXYTransfo()->worldToImage(worldPos.x(), worldPos.y(), imageI,
            imageJ); // there may be an issue with the conversion !!!

    QVector2D ijPt(imageI, imageJ);

    for (int i=0; i<m_discreatePolyLine.size(); i++) {
        QVector2D pt(m_discreatePolyLine[i]);
        if (pt.distanceToPoint(ijPt)<distance) {
            distance = pt.distanceToPoint(ijPt);
            indexNearestPoint = i;
        }
    }
    return std::tuple<long, double, bool>(indexNearestPoint, distance, true);
}
/*


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
						m_discreatePolyLine << newPt;
						double newPtXDouble, newPtYDouble;
						dataset->ijToXYTransfo()->imageToWorld(newPt.x(),newPt.y(), newPtXDouble, newPtYDouble);
						m_worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
					}
				}
			} else {
				for (long i=0; i<=dy;i++) {
					QPoint newPt;
					newPt.setY(A.y()+dirY*i);
					long addX = std::round(((double)(i*dx)) / dy);
					newPt.setX(A.x()+dirX*addX);
					if (newPt.x()>=0 && newPt.x()<nbXLine && newPt.y()>=0 && newPt.y()<nbInline) {
						m_discreatePolyLine << newPt;
						double newPtXDouble, newPtYDouble;
						dataset->ijToXYTransfo()->imageToWorld(newPt.x(), newPt.y(), newPtXDouble, newPtYDouble);
						m_worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
					}
				}
			}
 */


QVector3D RandomLineView::viewWorldTo3dWordExtended(QPointF posi)
{
	double worldX = posi.x();
	double worldY = posi.y();
	RandomRep *rep = firstRandom();
	if (rep == nullptr)
	{
		return QVector3D(0,0,0);
	}

	double realX, realY;
	Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
//	SeismicSurvey *survey = dataset->survey();

	QPointF ijPt;
	worldX = std::round(worldX); // round to get the point
	if (worldX>=0 && worldX<m_discreatePolyLine.size()) {
		long idx = worldX;
		ijPt = m_discreatePolyLine[idx];
	} else {
		int depassement = worldX - m_discreatePolyLine.size()+1;

		if(m_polyLine.size()<2) return QVector3D(0,0,0);

		QPolygon discreateNodes;

		for (std::size_t idx=m_polyLine.size()-2; idx<m_polyLine.size(); idx++) {
			double imageI, imageJ;
			dataset->ijToXYTransfo()->worldToImage(m_polyLine[idx].x(), m_polyLine[idx].y(),imageI, imageJ);
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
	dataset->ijToXYTransfo()->imageToWorld(ijPt.x(), ijPt.y(), realX, realY);


	return QVector3D(realX, realY, worldY);
}

QVector3D RandomLineView::viewWorldTo3dWord(QPointF posi)
{
	double worldX = posi.x();
	double worldY = posi.y();
	RandomRep *rep = firstRandom();
	if (rep == nullptr)
	{
		return QVector3D(0,0,0);
	}

	double realX, realY;
	Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
	SeismicSurvey *survey = dataset->survey();

	QPointF ijPt;
	worldX = std::round(worldX); // round to get the point
	if (worldX>=0 && worldX<m_discreatePolyLine.size()) {
		long idx = worldX;
		ijPt = m_discreatePolyLine[idx];
	} else {
		return QVector3D(0,0,0);
	}

	// there may be an issue with the conversion !!!
	dataset->ijToXYTransfo()->imageToWorld(ijPt.x(), ijPt.y(), realX, realY);


	return QVector3D(realX, realY, worldY);
}



QPointF RandomLineView::absoluteWorldToViewWorldExtended(QVector3D pos) {
    double worldX =pos.x();// event.worldX();
    double worldY = pos.z();//event.worldY();
  //  qDebug()<<"word to view worldX :"<<worldX<<" , worldY :"<<worldY;
    RandomRep *rep = firstRandom();
    if (rep == nullptr)
        return QPointF();

    double imageI, imageJ;

    Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
    SeismicSurvey *survey = dataset->survey();
    //    int pos = m_currentSliceWorldPosition;
    std::size_t indexNearestPoint;
    double distance = std::numeric_limits<double>::max();

    dataset->ijToXYTransfo()->worldToImage(worldX, worldY, imageI,
            imageJ); // there may be an issue with the conversion !!!

    QVector2D ijPt(imageI, imageJ);

    for (int i=0; i<m_discreatePolyLine.size(); i++) {
        QVector2D pt(m_discreatePolyLine[i]);
        if (pt.distanceToPoint(ijPt)<distance) {
            distance = pt.distanceToPoint(ijPt);
           // qDebug()<<i<<" distance proj :"<<distance;
            indexNearestPoint = i;
        }
    }

    std::pair<QPointF, QPointF> line;
    int indice=0;
    for (std::size_t idx=m_polyLine.size()-2; idx<m_polyLine.size(); idx++)
    {
    	double imageI, imageJ;
    	dataset->ijToXYTransfo()->worldToImage(m_polyLine[idx].x(), m_polyLine[idx].y(),imageI, imageJ);
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
				indexNearestPoint = m_discreatePolyLine.size()+ depas-1;
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
						indexNearestPoint = m_discreatePolyLine.size()+ depas-1;
					}
					else
					{
						//qDebug()<<" mindist2"<<dx<<" ," <<dy;
						int depas=  std::max(ceil(dx),ceil(dy));
						indexNearestPoint = m_discreatePolyLine.size()+ depas-1;
					}
				}

			}

    	}
    }

    long newPos = indexNearestPoint;
    return QPointF(newPos, pos.y());

}


QPointF RandomLineView::absoluteWorldToViewWorld(QVector3D pos) {
    double worldX =pos.x();// event.worldX();
    double worldY = pos.z();//event.worldY();
  //  qDebug()<<"word to view worldX :"<<worldX<<" , worldY :"<<worldY;
    RandomRep *rep = firstRandom();
    if (rep == nullptr)
        return QPointF();

    double imageI, imageJ;

    Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
    SeismicSurvey *survey = dataset->survey();
    //    int pos = m_currentSliceWorldPosition;
    std::size_t indexNearestPoint;
    double distance = std::numeric_limits<double>::max();

    dataset->ijToXYTransfo()->worldToImage(worldX, worldY, imageI,
            imageJ); // there may be an issue with the conversion !!!

  //  qDebug()<<m_discreatePolyLine.size()<<"  , image I: "<<imageI<<"  , image J: "<<imageJ;
    QVector2D ijPt(imageI, imageJ);

    for (int i=0; i<m_discreatePolyLine.size(); i++) {
        QVector2D pt(m_discreatePolyLine[i]);
        if (pt.distanceToPoint(ijPt)<distance) {
            distance = pt.distanceToPoint(ijPt);
           // qDebug()<<i<<" distance proj :"<<distance;
            indexNearestPoint = i;
        }
    }



    long newPos = indexNearestPoint;
    return QPointF(newPos, pos.y());
    /*if (event.hasDepth())
    {
        event.setPos(newPos, event.depth());
        return QPointF(newPos,event.depth())
    }
    else {
        double origin;
        dataset->sampleTransformation()->direct(0, origin);
        return QPointF(newPos, origin);
  }*/
  //  return true;
}



bool RandomLineView::absoluteWorldToViewWorld(MouseTrackingEvent &event) {
    double worldX = event.worldX();
    double worldY = event.worldY();

  //  qDebug()<<"worldX :"<<worldX<<" , worldY :"<<worldY;
    RandomRep *rep = firstRandom();
    if (rep == nullptr)
        return false;

    double imageI, imageJ;

    Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
    SeismicSurvey *survey = dataset->survey();
    //    int pos = m_currentSliceWorldPosition;
    std::size_t indexNearestPoint;
    double distance = std::numeric_limits<double>::max();

    dataset->ijToXYTransfo()->worldToImage(worldX, worldY, imageI,
            imageJ); // there may be an issue with the conversion !!!

    QVector2D ijPt(imageI, imageJ);

    for (int i=0; i<m_discreatePolyLine.size(); i++) {
        QVector2D pt(m_discreatePolyLine[i]);
        if (pt.distanceToPoint(ijPt)<distance) {
            distance = pt.distanceToPoint(ijPt);
            indexNearestPoint = i;
        }
    }
    //    if (m_viewType == ViewType::InlineView) {
    //        survey->inlineXlineToXYTransfo()->worldToImage(worldX, worldY, imageX,
    //                imageY);
    //        if (std::abs(pos - imageY) > 10)
    //            return false;
    //        newPos = imageX;
    //    } else if (m_viewType == ViewType::XLineView) {
    //        survey->inlineXlineToXYTransfo()->worldToImage(worldX, worldY, imageX,
    //                imageY);
    //        if (std::abs(pos - imageX) > 10)
    //            return false;
    //        newPos = imageY;
    //    }


    //    if ( distance> 10)
    //        return false;

    long newPos = indexNearestPoint;
    if (event.hasDepth())
        event.setPos(newPos, event.depth());
    else {
        double origin;
        dataset->sampleTransformation()->direct(0, origin);
        event.setPos(newPos, origin);
    }

    return true;
}

//void RandomLineView::getPositionMap(int position){
//    double positionX = position;
//    double positionY = 0;
//
//    RandomRep *rep = firstRandom();
//    if (rep != nullptr){
//        Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
//        SeismicSurvey *survey = dataset->survey();
//
//        QPointF ijPt;
//        positionX = std::round(positionX); // round to get the point
//        if (positionX>=0 && positionX<m_discreatePolyLine.size()) {
//            long idx = positionX;
//            ijPt = m_discreatePolyLine[idx];
//        }else{
//            return ;
//        }
//
//        // there may be an issue with the conversion !!!
//        dataset->ijToXYTransfo()->imageToWorld(ijPt.x(), ijPt.y(), positionX, positionY );
//    }
//    else{
//        return;
//    }
//
//    m_pointOthogonal = QPointF (positionX,positionY);
//}

void RandomLineView::getPositionMap(int position){
    if(m_eType == eTypeOrthogonal){
        double positionX = position;
        double positionY = 0;

        RandomRep *rep = firstRandom();
        if (rep != nullptr){
            Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
            SeismicSurvey *survey = dataset->survey();



            QPointF ijPt;
            positionX = std::round(positionX); // round to get the point
            if (positionX>=0 && positionX<m_discreateOrthogonalPolyLine.size()) {
                long idx = positionX;
                ijPt = m_discreateOrthogonalPolyLine[idx];
            }else{
                return ;
            }

            // there may be an issue with the conversion !!!
            dataset->ijToXYTransfo()->imageToWorld(ijPt.x(), ijPt.y(), positionX, positionY );
        }
        else{
            return;
        }

        m_pointOthogonal = QPointF (positionX,positionY);
        emit newPointPosition(m_pointOthogonal);

    }
}

bool RandomLineView::viewWorldToAbsoluteWorld(MouseTrackingEvent &event) {
    double worldX = event.worldX();
    double worldY = event.worldY();
    RandomRep *rep = firstRandom();
    if (rep == nullptr)
        return false;

    double realX, realY;
    Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) rep->data();
    SeismicSurvey *survey = dataset->survey();

    QPointF ijPt;
    worldX = std::round(worldX); // round to get the point
    if (worldX>=0 && worldX<m_discreatePolyLine.size()) {
        long idx = worldX;
        ijPt = m_discreatePolyLine[idx];
    } else {
        return false;
    }

    // there may be an issue with the conversion !!!
    dataset->ijToXYTransfo()->imageToWorld(ijPt.x(), ijPt.y(), realX, realY);

    //    int pos =m_currentSliceWorldPosition;
    //    if (m_viewType == ViewType::InlineView) {
    //        survey->inlineXlineToXYTransfo()->imageToWorld(worldX, pos, realX,
    //                realY);
    //    } else if (m_viewType == ViewType::XLineView) {
    //        survey->inlineXlineToXYTransfo()->imageToWorld(pos, worldX, realX,
    //                realY);
    //    }
    event.setPos(realX, realY, worldY, m_randomType);

    return true;
}


void RandomLineView::addAxis(IGeorefImage *image)
{
    if(m_verticalAxis == nullptr){
        m_verticalAxis = new QGLFixedAxisItem(image, VERTICAL_AXIS_SIZE, 5,QGLFixedAxisItem::Direction::VERTICAL);
        m_verticalAxisScene->addItem(m_verticalAxis);
        m_verticalAxisScene->setSceneRect(m_verticalAxis->boundingRect());
    }

    if(m_horizontalAxis == nullptr){
        m_horizontalAxis = new QGLFixedAxisItem(image, HORIZONTAL_AXIS_SIZE, 5,QGLFixedAxisItem::Direction::HORIZONTAL);
        m_horizontalAxisScene->addItem(m_horizontalAxis);
        m_horizontalAxisScene->setSceneRect(m_horizontalAxis->boundingRect());
    }
    updateAxisFromLengthUnit();

    if(m_eType == eTypeOrthogonal && m_crossItem==nullptr)
	{
    	m_rectIntersect =  image->worldExtent();
    	m_WidthOriginal = m_rectIntersect.width();

		int posX = image->worldExtent().left()+ image->worldExtent().width()*0.5f;
		int posY = image->worldExtent().top()+image->worldExtent().height()*0.5f;


		m_crossItem = new GraphEditor_CrossItem();
		m_crossItem->setZValue(CROSS_ITEM_Z);
		m_crossItem->setPos(QPointF(posX,posY));

		connect(m_crossItem, SIGNAL(signalAddPoints(QPointF)), this,SLOT(addFromCross(QPointF)));
		connect(m_crossItem, SIGNAL(signalPositionCross(int,QPointF)), this,SLOT(positionCross(int,QPointF)));
		connect(m_crossItem, SIGNAL(signalMoveCrossFinish()), this,SLOT(moveCrossFinish()));
		connect(this, SIGNAL(signalCurrentIndex(int)), m_crossItem,SLOT(setGrabberCurrent(int)));



		m_scene->addItem(m_crossItem);

		m_crossItem->show();

	}


}


void RandomLineView::addFromCross(QPointF pos)
{
	QVector3D pos3D  = viewWorldTo3dWordExtended(pos);
	QPointF posTr(pos3D.x(),pos3D.y());
	emit signalAddPointsDirectrice(posTr);

}

void RandomLineView::moveCross(int index,int dx,int dy)
{
	///QVector3D pos3D  = viewWorldTo3dWordExtended(pos);
	//QPointF posTr(pos3D.x(),pos3D.y());
	emit signalMovePointsDirectrice(index,dx,dy);

}

void RandomLineView::positionCross(int index,QPointF pos)
{
	QVector3D pos3D  = viewWorldTo3dWordExtended(pos);
	QPointF posTr(pos3D.x(),pos3D.y());
	emit signalPositionPointsDirectrice(index,posTr);

}

void RandomLineView::moveCrossFinish()
{

	emit signalMoveCrossFinish();

}

void RandomLineView::receiveCurrentGrabberIndex(int index)
{
	emit signalCurrentIndex(index);
}

//
//void RandomLineView::defineScale(RandomRep *rep) {
//    Seismic3DAbstractDataset *dst = (Seismic3DAbstractDataset*) rep->data();
//    double il, xl;
//    double x, y;
//    double x1, y1;
//
//    dst->ijToInlineXlineTransfo()->imageToWorld(0, 0, il, xl);
//    if (m_viewType == ViewType::RandomView) {
//        dst->survey()->inlineXlineToXYTransfo()->imageToWorld(xl, il, x, y);
//        dst->survey()->inlineXlineToXYTransfo()->imageToWorld(xl + 1, il, x1,
//                y1);
//
//        double scale = std::sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
//        m_scaleItem->setMapScale(scale);
//    }
//    /*else if (m_viewType == ViewType::XLineView) {
//        dst->survey()->inlineXlineToXYTransfo()->imageToWorld(xl, il, x, y);
//        dst->survey()->inlineXlineToXYTransfo()->imageToWorld(xl, il + 1, x1,
//                y1);
//
//        double scale = std::sqrt((x1 - x) * (x1 - x) + (y1 - y) * (y1 - y));
//        m_scaleItem->setMapScale(scale);
//    }*/
//}
//

void RandomLineView::removeAxis() {
    if (m_verticalAxis != nullptr) {
        m_verticalAxisScene->removeItem(m_verticalAxis);
        delete m_verticalAxis;
        m_verticalAxis = nullptr;
    }
    if (m_horizontalAxis != nullptr) {
        m_horizontalAxisScene->removeItem(m_horizontalAxis);
        delete m_horizontalAxis;
        m_horizontalAxis = nullptr;
    }
}

void RandomLineView::addOrthogonalLine(GraphEditor_LineShape* pLine){

	if(m_orthoLineList.size() ==7 )
	{
		GraphEditor_LineShape* first = m_orthoLineList[0];
		m_orthoLineList.clear();
		m_orthoLineList.push_back(first);

	}
    m_orthoLineList.push_back(pLine);
}

RandomLineView::~RandomLineView() {
    m_view->removeOverlayPainter(m_overlayPainter);
    delete m_overlayPainter;

	//qDebug()<<"destructeur de randomLineView";
	emit destroyed();
    for (GraphEditor_LineShape *r :m_orthoLineList){
       emit linedeteled(r);
    }

    if(m_ItemSection != nullptr){
        m_ItemSection->setRandomView(nullptr);
    }
}

/**
 * Contextual menu based on Graphics item
 */
void RandomLineView::contextualMenuFromGraphics(double worldX, double worldY, QContextMenuEvent::Reason reason, QMenu& menu) {
    //    QMenu menu("Seismic", m_view);
    m_contextualWorldX = worldX;
    m_contextualWorldY = worldY;

    QAction *showValuesAction = menu.addAction(("Show Values"));
    showValuesAction->setCheckable(m_isShowValues);
    connect(showValuesAction, SIGNAL(triggered(bool)), this,SLOT(showValues(bool)));

    //    QAction *actionSpectrum = menu.addAction(("Layer Spectrum Computation"));
    //    connect(actionSpectrum, SIGNAL(triggered()), this, SLOT(spectrumDecomposition()));

    //    QPoint mapPos = m_view->mapFromScene(QPointF( worldX, worldY));
    //    QPoint globalPos = m_view->mapToGlobal(mapPos);
    //    menu.exec(globalPos);
}

/**
 * Start show values on cursor move
 */
void RandomLineView::showValues(bool checked) {
    if (!m_isShowValues) {
        m_isShowValues = true;
    } else {
        m_isShowValues = false;
    }
}

void RandomLineView::discreatePoly(  Seismic3DAbstractDataset *dataset)
{
	long nbXLine = dataset->width();
	long nbInline = dataset->depth();

	QPolygon discreateNodes;
	m_discreatePolyLine.clear(); // just to be safe
	m_worldDiscreatePolyLine.clear(); // just to be safe
	for (std::size_t idx=0; idx<m_polyLine.size(); idx++) {
		double imageI, imageJ;
		dataset->ijToXYTransfo()->worldToImage(m_polyLine[idx].x(), m_polyLine[idx].y(),imageI, imageJ);
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
						m_discreatePolyLine << newPt;
						double newPtXDouble, newPtYDouble;
						dataset->ijToXYTransfo()->imageToWorld(newPt.x(),newPt.y(), newPtXDouble, newPtYDouble);
						m_worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
					}
				}
			} else {
				for (long i=0; i<=dy;i++) {
					QPoint newPt;
					newPt.setY(A.y()+dirY*i);
					long addX = std::round(((double)(i*dx)) / dy);
					newPt.setX(A.x()+dirX*addX);
					if (newPt.x()>=0 && newPt.x()<nbXLine && newPt.y()>=0 && newPt.y()<nbInline) {
						m_discreatePolyLine << newPt;
						double newPtXDouble, newPtYDouble;
						dataset->ijToXYTransfo()->imageToWorld(newPt.x(), newPt.y(), newPtXDouble, newPtYDouble);
						m_worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
					}
				}
			}
		}
	}
}

void RandomLineView::discreatePoly(QVector<QPointF> polyLine,  Seismic3DAbstractDataset *dataset,QPolygon& discreatePolyLine , QPolygonF& worldDiscreatePolyLine)
{
	long nbXLine = dataset->width();
	long nbInline = dataset->depth();

	QPolygon discreateNodes;
	discreatePolyLine.clear(); // just to be safe
	worldDiscreatePolyLine.clear(); // just to be safe
	for (std::size_t idx=0; idx<polyLine.size(); idx++) {
		double imageI, imageJ;
		dataset->ijToXYTransfo()->worldToImage(polyLine[idx].x(), polyLine[idx].y(),imageI, imageJ);
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
						discreatePolyLine << newPt;
						double newPtXDouble, newPtYDouble;
						dataset->ijToXYTransfo()->imageToWorld(newPt.x(),newPt.y(), newPtXDouble, newPtYDouble);
						worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
					}
				}
			} else {
				for (long i=0; i<=dy;i++) {
					QPoint newPt;
					newPt.setY(A.y()+dirY*i);
					long addX = std::round(((double)(i*dx)) / dy);
					newPt.setX(A.x()+dirX*addX);
					if (newPt.x()>=0 && newPt.x()<nbXLine && newPt.y()>=0 && newPt.y()<nbInline) {
						discreatePolyLine << newPt;
						double newPtXDouble, newPtYDouble;
						dataset->ijToXYTransfo()->imageToWorld(newPt.x(), newPt.y(), newPtXDouble, newPtYDouble);
						worldDiscreatePolyLine << QPointF(newPtXDouble, newPtYDouble);
					}
				}
			}
		}
	}
}



void RandomLineView::showRep(AbstractGraphicRep *rep) {
    bool isAddedCorrectly = true;
    QStringList errorMsg;
    RandomRep *slice = dynamic_cast<RandomRep*>(rep);

    //We need to add axis
    if (slice != nullptr) {
        if (firstRandom() == nullptr || (m_UpdateRep == true)) {
            //        defineScale(slice);

            if(m_UpdateRep == false){
               updateTile(slice->name());
            }
            m_UpdateRep = false;
            //            m_currentSliceIJPosition=slice->currentSliceIJPosition();
            //            m_currentSliceWorldPosition=slice->currentSliceWorldPosition();

            // extract
            Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) slice->data();

            discreatePoly(dataset);

            m_supportSurvey = dataset->survey();
            m_nbDatasetWidth = dataset->width();
            m_nbDatasetDepth = dataset->depth();
            isAddedCorrectly = true;
        } else {
            // remove incompatible reps
            Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) slice->data();
            long nbXLine = dataset->width();
            long nbInline = dataset->depth();
            if (nbXLine==m_nbDatasetWidth && nbInline==m_nbDatasetDepth && m_supportSurvey==dataset->survey()) {// isCompatible
                isAddedCorrectly = true;
            } else {
                isAddedCorrectly = false;// do not add
                errorMsg << "Incompatible dataset";
            }
        }
    } else {
        isAddedCorrectly = true;
    }
    ISampleDependantRep* sampleRep = dynamic_cast<ISampleDependantRep*>(rep);
    if (isAddedCorrectly && sampleRep!=nullptr) {
        QList<SampleUnit> units = sampleRep->getAvailableSampleUnits();
        if (m_randomType==SampleUnit::NONE && units.count()>0) {
            m_randomType = units[0];
            isAddedCorrectly = sampleRep->setSampleUnit(m_randomType);
        } else if (m_randomType!=SampleUnit::NONE && units.contains(m_randomType)) {
            isAddedCorrectly = sampleRep->setSampleUnit(m_randomType);
        } else{
            isAddedCorrectly = false;
        }
        if (!isAddedCorrectly && m_randomType!=SampleUnit::NONE) {
            errorMsg << sampleRep->getSampleUnitErrorMessage(m_randomType);
        } else if (!isAddedCorrectly) {
            errorMsg << "Display unit unknown";
        }
    }

    if (isAddedCorrectly) {
        Abstract2DInnerView::showRep(rep);
        updateTitleFromSlices();
    } else{
        // fail to add
        //qDebug() << "RandomLineView : fail to add rep " << rep->name() << " error messages : "<< errorMsg;
    }

    if((slice != nullptr) && (isAddedCorrectly == true ) && (slice->image() != nullptr)){
        if(m_eType != eTypeOrthogonal){
            m_slice = slice;
            int departX = slice->image()->worldExtent().x();
            int width = slice->image()->worldExtent().width();
            defineSliceMinMax(departX, width);//lice->image());
        }
        addAxis(slice->image());


    }

    //    for (AbstractGraphicRep *r : m_visibleReps) {
    //        RandomRep* pSliceRep = dynamic_cast<RandomRep*>(r);
    //        if(pSliceRep != nullptr){
    //
    //            if(pSliceRep->name().toLower().contains("rgt")){
    //                hideRep(r);
    //                Abstract2DInnerView::showRep(r);
    //                if(pSliceRep->image() != nullptr){
    //                    addAxis(pSliceRep->image());
    //                }
    //                break;
    //            }
    //        }
    //    }
}

void RandomLineView::hideRep(AbstractGraphicRep *rep) {
    Abstract2DInnerView::hideRep(rep);
    if (firstRandom() == nullptr) {
        removeAxis();
        m_discreatePolyLine.clear();
        m_worldDiscreatePolyLine.clear();
        //m_supportSurvey = nullptr;
    }
    if (m_visibleReps.count()==0) {
        m_randomType = SampleUnit::NONE;
    }
    updateTitleFromSlices();
}

void RandomLineView::cleanupRep(AbstractGraphicRep *rep) {
    Abstract2DInnerView::cleanupRep(rep);
    if (firstRandom() == nullptr) {
        removeAxis();
        m_discreatePolyLine.clear();
        m_worldDiscreatePolyLine.clear();
        m_supportSurvey = nullptr;
    }
    if (m_visibleReps.count()==0) {
        m_randomType = SampleUnit::NONE;
    }
    updateTitleFromSlices();
}

double RandomLineView::displayDistance() const {
    return m_displayDistance;
}

void RandomLineView::setDisplayDistance(double val) {
    if (m_displayDistance!=val) {
        m_displayDistance = val;
        emit displayDistanceChanged(m_displayDistance);
    }
}
;

void RandomLineView::updateTitleFromSlices() {
	if (m_suffixTitle.isEmpty() || m_suffixTitle.isNull()) {
		// no name, search for it
		QString rgtName;
		RandomRep* sliceRgtObj = nullptr;
		QString name;
		RandomRep* sliceObj = nullptr;
		int i = 0;
		while ((name.isNull() || name.isEmpty()) && i<m_visibleReps.size()) {
			AbstractGraphicRep *r = m_visibleReps[i];
			RandomRep *slice = dynamic_cast<RandomRep*>(r);
			if (slice && slice->data()) {
				Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(slice->data());
				if (dataset && dataset->type()!=Seismic3DAbstractDataset::RGT) {
					name = dataset->name();
					sliceObj = slice;
				} else if (dataset && (rgtName.isNull() || rgtName.isEmpty())) {
					rgtName = dataset->name();
					sliceRgtObj = slice;
				}
			}
			i++;
		}
		if ((name.isNull() || name.isEmpty()) && !rgtName.isNull() && !rgtName.isEmpty()) {
			name = rgtName;
			m_cacheIsNameRgt = true;
			m_cacheSliceNameObj = sliceRgtObj;
			updateTile(name);
		} else if (!name.isNull() && !name.isEmpty()) {
			m_cacheIsNameRgt = false;
			m_cacheSliceNameObj = sliceObj;
			updateTile(name);
		} else {
			updateTile("");
		}
	} else if (!m_cacheIsNameRgt) {
		// search if name holder is still there
		bool nameHolderPresent = false;
		int i = 0;
		while (!nameHolderPresent && i<m_visibleReps.size()) {
			nameHolderPresent = m_visibleReps[i]==m_cacheSliceNameObj;
			i++;
		}
		if (!nameHolderPresent) {
			updateTile("");
			updateTitleFromSlices();
		}
	} else {
		bool nameHolderPresent = false;
		bool foundSeismic = false;
		int i = 0;
		while ((!nameHolderPresent || !foundSeismic) && i<m_visibleReps.size()) {
			if (!nameHolderPresent) {
				nameHolderPresent = m_visibleReps[i]==m_cacheSliceNameObj;
			}
			if (!foundSeismic) {
				RandomRep* slice = dynamic_cast<RandomRep*>(m_visibleReps[i]);
				bool valid = slice!=nullptr;
				if (valid) {
					Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(slice->data());
					valid = dataset->type()!=Seismic3DAbstractDataset::RGT;
				}
				foundSeismic = valid;
			}
			i++;
		}
		if (!nameHolderPresent || foundSeismic) {
			updateTile("");
			updateTitleFromSlices();
		}
	}
}

void RandomLineView::setDepthLengthUnitProtected(const MtLengthUnit* depthLengthUnit) {
	Abstract2DInnerView::setDepthLengthUnitProtected(depthLengthUnit);

	updateAxisFromLengthUnit();
}

void RandomLineView::updateAxisFromLengthUnit() {
	double a;
	if (m_randomType==SampleUnit::DEPTH) {
		a = MtLengthUnit::convert(MtLengthUnit::METRE, *m_depthLengthUnit, 1);
	} else {
		a = 1;
	}
	AffineTransformation transfo(a, 0);
	if (m_verticalAxis) {
		m_verticalAxis->setDisplayValueTransform(&transfo);
		m_verticalAxisScene->update(m_verticalAxisScene->sceneRect());
	}
}

void RandomLineView::mainSceneRectChanged() {
	if (m_verticalAxisScene) {
		QRectF sceneRect = m_verticalAxisScene->sceneRect();
		sceneRect.setY(m_scene->sceneRect().y());
		sceneRect.setHeight(m_scene->sceneRect().height());
		m_verticalAxisScene->setSceneRect(sceneRect);
	}
	if (m_horizontalAxisScene) {
		QRectF sceneRect = m_horizontalAxisScene->sceneRect();
		sceneRect.setX(m_scene->sceneRect().x());
		sceneRect.setWidth(m_scene->sceneRect().width());
		m_horizontalAxisScene->setSceneRect(sceneRect);
	}
}


void RandomLineView::moveLineOrtho(QPointF position ,QPointF normal)
{
	//qDebug()<<"==> m_orthoLineList.count() " << m_orthoLineList.count();

	if( m_orthoLineList.count()>0 )
	{
		for(int i=0;i<m_orthoLineList.count() ;i++)
		{
			if(m_orthoLineList[i] != nullptr)
			{
				QVector3D nor3d(normal.x(),0.0f,normal.y());
				m_orthoLineList[i]->refreshOrtholine(nor3d, position);
			}
		}
	}
}

void RandomLineView::initTransformation()
{
	RandomRep *slice = firstRandom();
	 if (slice != nullptr)
	 {
		 Seismic3DAbstractDataset *dataset = (Seismic3DAbstractDataset*) slice->data();

		 m_transformation = new RandomTransformation( dataset->height(), m_polyLine, *dataset->ijToXYTransfo(),this);

	 }

}

RandomTransformation* RandomLineView::randomTransform()
{
	return m_transformation;
}

double RandomLineView::getWellSectionWidth() const {
	return m_wellSectionWidth;
}

void RandomLineView::setWellSectionWidth(double value) {
	if(m_wellSectionWidth != value) {
		m_wellSectionWidth = value;
		emit signalWellSectionWidth(m_wellSectionWidth);
	}
}


void RandomLineView::resetItemSection()
{
	m_ItemSection=nullptr;
}



